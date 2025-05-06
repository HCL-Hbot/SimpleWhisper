#ifndef WHISPER_NODE_HPP
#define WHISPER_NODE_HPP

#include <chrono>
#include <csignal>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "audio_tools/msg/audio_data.hpp"
#include "audio_tools/msg/audio_data_stamped.hpp"
#include "audio_tools/msg/audio_info.hpp"

#include "audio_async_ros/audio_async_ros.hpp"
#include "simplewhisper.hpp"

class WhisperNode : public rclcpp::Node {
public:
  WhisperNode();
  ~WhisperNode();

private:
  // === Audio & Processing ===
  void audio_cb(const audio_tools::msg::AudioDataStamped::SharedPtr msg);
  void processAudio();

  // === VAD and Filtering (used internally) ===
  bool vad_simple(std::vector<float> &pcmf32, int sample_rate, int last_ms,
                  float vad_thold, float freq_thold, bool verbose);
  void high_pass_filter(std::vector<float> &data, float cutoff, float sample_rate);

  // === ROS Interfaces ===
  rclcpp::Subscription<audio_tools::msg::AudioDataStamped>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;

  // === Core Components ===
  std::shared_ptr<audio_async> audio_;
  std::shared_ptr<SimpleWhisper> whisper_;

  // === Worker Thread Control ===
  std::thread worker_thread_;
  std::atomic<bool> keep_running_;

  // === Parameters ===
  float vad_threshold_;
  float freq_threshold_;
  int voice_duration_ms_;
};

#endif // WHISPER_NODE_HPP
