#ifndef WHISPER_NODE_HPP
#define WHISPER_NODE_HPP
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "audio_tools/msg/audio_data.hpp"
#include "audio_tools/msg/audio_data_stamped.hpp"
#include "audio_tools/msg/audio_info.hpp"

#include "audio_async_ros/audio_async_ros.hpp"
#include "simplewhisper.hpp"

#include <chrono>
#include <csignal>
#include <memory>
#include <thread>
#include <vector>

class WhisperNode : public rclcpp::Node {
    public:
      WhisperNode();

      ~WhisperNode();
      private:
      /**
       * @brief Callback function for received audio_msg on subscribed topic
       * @param msg The audio message which got received
       */
      void audio_cb(const audio_tools::msg::AudioDataStamped::SharedPtr msg);
      bool vad_simple(std::vector<float> &pcmf32, int sample_rate, int last_ms,
        float vad_thold, float freq_thold, bool verbose);
      void high_pass_filter(std::vector<float> &data, float cutoff, float sample_rate);
      void processAudio();
  /**
   * @brief Subscription object/handle for the subscribed audio topic
   */
  rclcpp::Subscription<audio_tools::msg::AudioDataStamped>::SharedPtr
      subscription_;
  std::shared_ptr<audio_async> audio_;
  std::shared_ptr<SimpleWhisper> whisper_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  std::thread worker_thread_;
  std::atomic<bool> keep_running_;
  float vad_threshold_;
  float freq_threshold_;
  int voice_duration_ms_;
};
#endif /* WHISPER_NODE_HPP */