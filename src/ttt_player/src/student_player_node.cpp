#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

const std::vector<std::pair<uint8_t, uint8_t>> kScriptedMoves = {
    {6, 2},
    {7, 4},
    {8, 6},
};

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  void try_register() {
    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ----------------------------------------------------------------
  // IK helpers
  // ----------------------------------------------------------------

  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {
    (void)target_pose;
    (void)seed_positions;

    // TODO(student): Call the MoveIt `/compute_ik` service here.
    // Suggested steps:
    // 1. Create a `moveit_msgs::srv::GetPositionIK::Request`.
    // 2. Set `group_name = "panda_arm"`.
    // 3. Fill the seed joint state with the provided `seed_positions`.
    // 4. Set the target pose in frame `panda_link0`.
    // 5. Send the request through `ik_client_` and wait for the response.
    // 6. Extract the 7 Panda arm joints from the solution and return them.
    // 7. Return `std::nullopt` if IK times out or fails.
    //
    // The dummy return below keeps the starter code buildable, but it does not
    // solve IK. Students should replace it with a real implementation.
   
    //以下todo代码

 

    // 1
    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name = "panda_arm";          
    request->ik_request.pose_stamped.header.frame_id = "panda_link0"; 
    request->ik_request.pose_stamped.header.stamp = this->now();
    request->ik_request.pose_stamped.pose = target_pose;      

 

    // 2

    request->ik_request.robot_state.joint_state.name = panda_joint_names();
    request->ik_request.robot_state.joint_state.position = seed_positions;

 

    // 

    auto future = ik_client_->async_send_request(request);
      if (future.wait_for(1s) != std::future_status::ready) {
      RCLCPP_WARN(this->get_logger(), "IK request timed out.");
      return std::nullopt;
    }

      auto response = future.get();
      if (response->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        RCLCPP_WARN(this->get_logger(), "IK failed with code %d", response->error_code.val);
        return std::nullopt;


      // 6      
      const auto &solution = response->solution.joint_state;
      std::vector<double> result;
      result.reserve(panda_joint_names().size());
      for (const auto &joint_name : panda_joint_names()) {
        auto it = std::find(solution.name.begin(), solution.name.end(), joint_name);
        if (it == solution.name.end()) {
          RCLCPP_ERROR(this->get_logger(), "Joint %s missing in IK solution.", joint_name.c_str());
          return std::nullopt;
        }
        size_t idx = std::distance(solution.name.begin(), it);
        result.push_back(solution.position[idx]);
      }
      return result;

      
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "IK request exception: %s", e.what());
      return std::nullopt;
    }

  //////////////以上todo代码
  }

  // ----------------------------------------------------------------
  // Turn planning
  // ----------------------------------------------------------------

  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    // TODO(student): Implement your turn-planning logic here.
    // Suggested structure:
    // 1. Choose a legal `(piece_id, cell_id)` pair from `request->snapshot`.
    // 2. Look up the current pose of the chosen stock piece.
    // 3. Look up the target board cell pose from `request->layout.cell_poses`.
    // 4. Convert those TCP targets into `panda_link8` poses using
    //    `link8_pose_from_tcp_target(...)`.
    // 5. Call `compute_ik(...)` for the pick target and place target.
    // 6. Build the four required trajectories:
    //      - home_to_pick
    //      - pick_to_home
    //      - home_to_place
    //      - place_to_home
    // 7. Fill `response->plan` and set `response->accepted = true` on success.
    //
    // The fallback below intentionally rejects every turn. This keeps the
    // starter repository buildable while making it clear that students must
    // implement their own planner.
    
    //以下todo代码

    // 1
    if (request->snapshot.legal_actions.empty()) {
      response->accepted = false;
      response->message = "No legal actions available.";
      return;
    }
    const auto &action = request->snapshot.legal_actions.front();
    uint8_t piece_id = action.piece_id;
    uint8_t cell_id = action.cell_id;

 

    // 2
    geometry_msgs::msg::Pose piece_pose = find_piece_pose(request->snapshot, piece_id);

 

    // 
    if (cell_id >= request->layout.cell_poses.size()) {
      response->accepted = false;
      response->message = "Invalid cell id.";
      return;
    }
    const auto &cell_pose = request->layout.cell_poses[cell_id];

 

    // 4
    auto pick_pose = link8_pose_from_tcp_target(
        piece_pose.position.x, piece_pose.position.y, piece_pose.position.z);
    auto place_pose = link8_pose_from_tcp_target(
        cell_pose.position.x, cell_pose.position.y, cell_pose.position.z);

 

    // 5.
    auto pick_joints = compute_ik(pick_pose, kHomePositions);
    auto place_joints = compute_ik(place_pose, kHomePositions);
    if (!pick_joints || !place_joints) {
      response->accepted = false;
      response->message = "IK failed for pick/place.";
      return;
    }

 

    //

    ttt_interfaces::msg::TurnPlan plan;
    plan.home_to_pick = make_three_point_trajectory(kHomePositions, *pick_joints, 2.0);
    plan.pick_to_home = make_three_point_trajectory(*pick_joints, kHomePositions, 2.0);
    plan.home_to_place = make_three_point_trajectory(kHomePositions, *place_joints, 2.0);
    plan.place_to_home = make_three_point_trajectory(*place_joints, kHomePositions, 2.0);


    response->plan = plan;
    response->accepted = true;
    response->message = "Plan generated successfully.";
 

  } catch (const std::exception &e) {
    response->accepted = false;
    response->message = std::string("Exception in handle_plan_turn: ") + e.what();

 

  ///以上todo代码

    //response->accepted = false;
    //response->message = "TODO(student): implement handle_plan_turn().";

    //以上两行原有的
  }

  static geometry_msgs::msg::Pose find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      uint8_t piece_id) {
    // TODO(student): Search `snapshot.pieces` for the requested `piece_id` and
    // return its pose. You may choose to throw an exception or return a
    // fallback pose if the piece is missing.

    //以下两行原有的占位

    //(void)snapshot;
    //(void)piece_id;

 

  ///以下todo代码

    for (const auto &piece : snapshot.pieces) {
      if (piece.piece_id == piece_id) {
        return piece.pose;
      }
    }


    RCLCPP_WARN(rclcpp::get_logger("student_player"),
                "Piece %u not found in snapshot, returning fallback pose.",  piece_id);

  ///以上todo代码

    // Dummy fallback to keep the starter code compilable.
    geometry_msgs::msg::Pose fallback;
    fallback.orientation.w = 1.0;
    return fallback;
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
