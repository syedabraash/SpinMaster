# **SpinMaster: Intelligent Table Tennis Analysis**

## **Overview**  
SpinMaster is an AI-powered application designed to analyze table tennis gameplay for players seeking to improve their performance. By leveraging computer vision and machine learning models, SpinMaster provides detailed feedback on player movements, ball trajectories, and shot accuracies. This project is inspired by advanced analytics systems used in professional sports and aims to make performance improvement accessible to table tennis enthusiasts.

---

## **Features**
- **Player Movement Detection**: Using the MoveNet Pose Estimation model from TensorFlow Hub, the system accurately detects player movements from side-angle match footage (more angles to be tested with time).
- **Game Footage Analysis**: SpinMaster currently supports clips from the Saudi Smash tournament (all credits to wtt), particularly the match between Wang Chuqin and Patrick Franziska.
- **Data-Driven Feedback**: The system offers insights into player positioning, reaction times, and movement efficiency.

---

## **Current Development Status**
- **Working Features**:
  - **Player Pose Estimation**: The MoveNet Pose Estimation model detects player movements with high precision.
  - **Compatible with Low-MB Files**: The system can process low-MB video files efficiently.
  
- **Ongoing Work**:
  - **Ball Detection**: The current ball detection module is under development and is not yet functional. Challenges include:
    - Difficulty detecting the ball's movement in real-time.
    - Noisy data leading to inaccurate predictions.

- **Known Limitations**:
  - The system struggles to process larger video files due to memory constraints, causing crashes or hangs.

---

## **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- FFmpeg (for video preprocessing)

### **Installation**
1. Clone this repository:  
   ```bash
   git clone https://github.com/syedabraash/spinmaster.git
   cd spinmaster
   ```
2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

### **Usage**
1. Place your video files in the `input_videos/` directory.
2. Run the main script:  
   ```bash
   python spinmaster.py
   ```
3. Processed results will be saved in the `output/` directory, including annotated videos and data summaries.

---

## **Model Information**
- **Player Movement Detection**:  
  The MoveNet Pose Estimation model from TensorFlow Hub is used to detect player poses with high accuracy from side-angle video coverage.

- **Supported Datasets**:  
  - The system has been tested using match footage from the Saudi Smash tournament.

---

## **Roadmap**
- **Short-Term Goals**:
  - Improve ball detection algorithms.
  - Optimize system performance for larger video files.
  
- **Long-Term Goals**:
  - Add shot classification (e.g., forehand, backhand, serve).
  - Incorporate ball spin and speed analysis.
  - Develop a user-friendly GUI for easier interaction.

---

## **Known Issues**
- The system currently crashes or hangs when processing large video files.
- Ball detection module remains non-functional as of now.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the coding standards outlined in the CONTRIBUTING.md file.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Acknowledgments**
- **TensorFlow Hub** for the MoveNet Pose Estimation model.
- Footage from the **Saudi Smash Tournament by WTT**, used for testing and development.
- Inspiration from table tennis analytics systems and sports data analysis projects.
