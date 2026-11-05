// Keyboard-only Unitree G1 high-level preparation and handoff.
//
// This program does not start SuperMimic.  It exposes the vendor high-level
// state machine first, then releases it only after the operator presses `g`
// while the robot is demonstrably stable.  The wrapper script starts the
// low-level SuperMimic runner after this process exits successfully.

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

namespace {
using Clock = std::chrono::steady_clock;
using LowState = unitree_hg::msg::dds_::LowState_;
std::atomic_bool stop_requested{false};

void on_signal(int) { stop_requested = true; }

class Terminal {
 public:
  Terminal() {
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved_) != 0)
      throw std::runtime_error("keyboard handoff requires an interactive TTY");
    auto mode = saved_;
    mode.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    mode.c_cc[VMIN] = 0;
    mode.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &mode) != 0)
      throw std::runtime_error("cannot configure terminal");
  }
  ~Terminal() { tcsetattr(STDIN_FILENO, TCSANOW, &saved_); }
  Terminal(const Terminal&) = delete;
  Terminal& operator=(const Terminal&) = delete;
 private:
  termios saved_{};
};

struct Snapshot {
  bool received = false;
  Clock::time_point updated{};
  int fsm_code = -1;
  int fsm = -1;
  float roll = 0.0F;
  float pitch = 0.0F;
  float max_dq = 0.0F;
  int max_temp = 0;
  bool motor_error = false;
};

class Robot {
 public:
  explicit Robot(const std::string& iface) : sub_("rt/lowstate") {
    loco_.SetTimeout(0.5F);
    loco_.Init();
    switcher_.SetTimeout(0.5F);
    switcher_.Init();
    sub_.InitChannel([this](const void* raw) {
      if (!raw) return;
      const auto& s = *static_cast<const LowState*>(raw);
      std::lock_guard<std::mutex> lock(mu_);
      snap_.received = true;
      snap_.updated = Clock::now();
      const auto& rpy = s.imu_state().rpy();
      snap_.roll = rpy[0];
      snap_.pitch = rpy[1];
      snap_.max_dq = 0.0F;
      snap_.max_temp = 0;
      snap_.motor_error = false;
      for (int i = 0; i < 29; ++i) {
        const auto& m = s.motor_state()[i];
        snap_.max_dq = std::max(snap_.max_dq, std::abs(m.dq()));
        snap_.max_temp = std::max(snap_.max_temp, static_cast<int>(m.temperature()[0]));
        snap_.max_temp = std::max(snap_.max_temp, static_cast<int>(m.temperature()[1]));
        snap_.motor_error = snap_.motor_error || m.motorstate() != 0;
      }
    }, 1);
  }

  ~Robot() { sub_.CloseChannel(); }

  Snapshot snapshot() {
    Snapshot out;
    {
      std::lock_guard<std::mutex> lock(mu_);
      out = snap_;
    }
    int fsm = -1;
    const int fsm_code = loco_.GetFsmId(fsm);
    out.fsm_code = fsm_code;
    if (fsm_code == 0) {
      out.fsm = fsm;
    }
    return out;
  }

  int zero() { return loco_.ZeroTorque(); }
  int damp() { return loco_.Damp(); }
  int stand() { return loco_.StandUp(); }
  int start() { return loco_.Start(); }
  int balance() { return loco_.BalanceStand(); }

  int check_mode(std::string& form, std::string& name) {
    return switcher_.CheckMode(form, name);
  }

  int release() { return switcher_.ReleaseMode(); }

 private:
  std::mutex mu_;
  Snapshot snap_;
  unitree::robot::g1::LocoClient loco_;
  unitree::robot::b2::MotionSwitcherClient switcher_;
  unitree::robot::ChannelSubscriber<LowState> sub_;
};

bool healthy(const Snapshot& s) {
  return s.received && (Clock::now() - s.updated < std::chrono::milliseconds(500)) &&
         std::abs(s.roll) < 0.35F && std::abs(s.pitch) < 0.35F &&
         s.max_dq < 1.0F && s.max_temp < 80 && !s.motor_error;
}

void print_status(Robot& robot) {
  const auto s = robot.snapshot();
  std::cout << "fsm=" << s.fsm << " (code=" << s.fsm_code << ")"
            << " roll=" << s.roll << " pitch=" << s.pitch
            << " max_dq=" << s.max_dq << " max_temp=" << s.max_temp
            << " healthy=" << (healthy(s) ? "yes" : "no") << std::endl;
}

void help() {
  std::cout << "High-level keyboard phase (no SuperMimic commands yet)\n"
            << "z: ZeroTorque (FSM 0; support required)\n"
            << "p: Damp (FSM 1; does not hold standing posture)\n"
            << "u: StandUp (FSM 4)\n"
            << "t: Start / enter FSM 500\n"
            << "b: BalanceStand request (must return 0)\n"
            << "i: print status\n"
            << "g: release high-level mode and hand off to SuperMimic (only when stable)\n"
            << "q/Ctrl-C: exit without handoff\n" << std::flush;
}
}  // namespace

int main(int argc, char** argv) {
  std::string iface = "eth0";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--interface" && i + 1 < argc) iface = argv[++i];
    else if (std::string(argv[i]) == "--help") { help(); return 0; }
    else { std::cerr << "usage: g1_highlevel_keyboard [--interface eth0]\n"; return 2; }
  }
  for (int sig : {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGTSTP}) std::signal(sig, on_signal);
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, iface);
    Robot robot(iface);
    Terminal terminal;
    help();
    bool handed_off = false;
    bool balance_requested = false;
    while (!stop_requested) {
      pollfd p{STDIN_FILENO, POLLIN, 0};
      if (poll(&p, 1, 100) <= 0 || !(p.revents & POLLIN)) continue;
      char c = 0;
      if (read(STDIN_FILENO, &c, 1) != 1) break;
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (c == 'q' || c == 3) break;
      if (c == 'i') { print_status(robot); continue; }
      int code = 0;
      if (c == 'z') { balance_requested = false; code = robot.zero(); }
      else if (c == 'p') { balance_requested = false; code = robot.damp(); }
      else if (c == 'u') { balance_requested = false; code = robot.stand(); }
      else if (c == 't') { balance_requested = false; code = robot.start(); }
      else if (c == 'b') {
        code = robot.balance();
        balance_requested = code == 0;
      }
      else if (c == 'g') {
        // The G1 firmware may not expose GetBalanceMode (error 7301), so use
        // the successful BalanceStand request plus independent health checks.
        Snapshot s;
        bool ready = false;
        for (int attempt = 0; attempt < 30 && !stop_requested; ++attempt) {
          s = robot.snapshot();
          ready = balance_requested && s.fsm == 500 && healthy(s);
          if (ready) break;
          if (attempt == 0 || attempt == 10 || attempt == 20) {
            std::cout << "waiting for stable balance: fsm=" << s.fsm
                      << " (code=" << s.fsm_code << ") balance_request="
                      << (balance_requested ? "ok" : "missing")
                      << " healthy=" << (healthy(s) ? "yes" : "no") << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!ready) {
          std::cout << "handoff blocked: require FSM=500, accepted BalanceStand, stable healthy state"
                    << " (balance request must return 0; FSM code=" << s.fsm_code << ")"
                    << std::endl;
          continue;
        }
        std::string form, name;
        const int check = robot.check_mode(form, name);
        std::cout << "CheckMode code=" << check << " form=" << form << " name=" << name << std::endl;
        if (check != 0 || name.empty()) {
          std::cout << "handoff blocked: high-level mode is not active" << std::endl;
          continue;
        }
        std::cout << "Releasing high-level mode; keep the hoist ready." << std::endl;
        code = robot.release();
        if (code == 0) { handed_off = true; break; }
      } else continue;
      std::cout << "request " << c << " return_code=" << code << std::endl;
    }
    if (handed_off) {
      std::cout << "HANDOFF_OK: high-level mode released; start the SuperMimic runner now." << std::endl;
      return 0;
    }
    std::cout << "Exited without SuperMimic handoff." << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << std::endl;
    return 1;
  }
}
