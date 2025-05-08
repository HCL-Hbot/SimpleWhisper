#include "whisper_node.hpp"

#define AUDIO_DEFAULT_MIC -1
#define AUDIO_SAMPLE_RATE 16000
constexpr int AUDIO_BUFFER_SIZE_MS = 2000;
constexpr int VAD_LOOKBACK_MS = 1250;
constexpr int QUEUE_SIZE = 10;

WhisperNode::WhisperNode() : Node("whisper_node"), keep_running_(true) {
  // Declare parameters
  declare_parameter<std::string>("model_path", "ggml-large-v3-turbo-q5_0.bin");
  declare_parameter<std::string>("language", "en");
  declare_parameter<std::string>("bot_name", "Llama");
  declare_parameter<std::string>("output_topic", "whisper/text");
  declare_parameter<std::string>("audio_in_topic", "whisper/audio_in");
  declare_parameter<bool>("translate", false);
  declare_parameter<float>("vad_threshold", 0.6f);
  declare_parameter<float>("freq_threshold", 100.0f);
  declare_parameter<int>("voice_duration_ms", 10000);

  // Read and assign thresholds and timing
  vad_threshold_ = get_parameter("vad_threshold").as_double();
  freq_threshold_ = get_parameter("freq_threshold").as_double();
  voice_duration_ms_ = get_parameter("voice_duration_ms").as_int();

  // Whisper setup
  simplewhisper_model_params model_params;
  model_params.model_wsp = get_parameter("model_path").as_string();
  model_params.language = get_parameter("language").as_string();
  model_params.bot_name = get_parameter("bot_name").as_string();
  model_params.translate = get_parameter("translate").as_bool();

  whisper_ = std::make_shared<SimpleWhisper>(model_params);
  whisper_->init();

  // Audio setup
  audio_ = std::make_shared<audio_async>(20000);
  audio_->init(AUDIO_DEFAULT_MIC, AUDIO_SAMPLE_RATE);
  audio_->resume();
  audio_->clear();

  // Publisher
  const std::string output_topic = get_parameter("output_topic").as_string();
  pub_ =
      this->create_publisher<std_msgs::msg::String>(output_topic, QUEUE_SIZE);

  // Subscriber
  const std::string audio_topic = get_parameter("audio_in_topic").as_string();
  subscription_ = this->create_subscription<audio_tools::msg::AudioDataStamped>(
      audio_topic, QUEUE_SIZE,
      std::bind(&WhisperNode::audio_cb, this, std::placeholders::_1));

  // Start background worker thread
  worker_thread_ = std::thread(&WhisperNode::processAudio, this);
}

WhisperNode::~WhisperNode() {
  keep_running_ = false;
  if (worker_thread_.joinable())
    worker_thread_.join();
}

/**
 * @brief Callback function for received audio_msg on subscribed topic
 * @param msg The audio message which got received
 */
void WhisperNode::audio_cb(
    const audio_tools::msg::AudioDataStamped::SharedPtr msg) {
  const auto &info = msg->info;
  const auto &audio = msg->audio;

  if (audio.data.empty())
    return;
  // Convert int16_t audio to float [-1.0, 1.0]
  size_t n_samples = audio.data.size() / sizeof(int16_t);
  const int16_t *raw_samples =
      reinterpret_cast<const int16_t *>(audio.data.data());

  std::vector<float> float_audio(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    float_audio[i] = static_cast<float>(raw_samples[i]) / 32768.0f;
  }
  audio_->callback(reinterpret_cast<const uint8_t *>(float_audio.data()),
                   float_audio.size() * sizeof(float));
}

void WhisperNode::high_pass_filter(std::vector<float> &data, float cutoff,
                                   float sample_rate) {
  const float rc = 1.0f / (2.0f * M_PI * cutoff);
  const float dt = 1.0f / sample_rate;
  const float alpha = dt / (rc + dt);

  float y = data[0];

  for (size_t i = 1; i < data.size(); i++) {
    y = alpha * (y + data[i] - data[i - 1]);
    data[i] = y;
  }
}

bool WhisperNode::vad_simple(std::vector<float> &pcmf32, int sample_rate,
                             int last_ms, float vad_thold, float freq_thold,
                             bool verbose) {
  const int n_samples = pcmf32.size();
  const int n_samples_last = (sample_rate * last_ms) / 1000;

  if (n_samples_last >= n_samples) {
    // not enough samples - assume no speech
    return false;
  }

  if (freq_thold > 0.0f) {
    high_pass_filter(pcmf32, freq_thold, sample_rate);
  }

  float energy_all = 0.0f;
  float energy_last = 0.0f;

  for (int i = 0; i < n_samples; i++) {
    energy_all += fabsf(pcmf32[i]);

    if (i >= n_samples - n_samples_last) {
      energy_last += fabsf(pcmf32[i]);
    }
  }

  energy_all /= n_samples;
  energy_last /= n_samples_last;

  if (verbose) {
    fprintf(
        stderr,
        "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n",
        __func__, energy_all, energy_last, vad_thold, freq_thold);
  }

  if (energy_last > vad_thold * energy_all) {
    return false;
  }

  return true;
}

void WhisperNode::processAudio() {
  std::vector<float> buffer;

  while (rclcpp::ok() && keep_running_) {
    audio_->get(AUDIO_BUFFER_SIZE_MS, buffer);

    bool voice_activity_detected =
        vad_simple(buffer, WHISPER_SAMPLE_RATE, VAD_LOOKBACK_MS, vad_threshold_,
                   freq_threshold_, false);

    if (voice_activity_detected) {
      RCLCPP_INFO(this->get_logger(), "Voice detected");
      buffer.clear();
      audio_->get(voice_duration_ms_, buffer);
      std::string result = whisper_->do_inference(buffer);

      if (!result.empty()) {
        std_msgs::msg::String msg;
        msg.data = result;
        pub_->publish(msg);
      }
      audio_->clear();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<WhisperNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
