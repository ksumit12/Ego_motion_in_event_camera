# 15-Minute Presentation Script: Ego Motion Compensation in Event Cameras

## **Slide 1: Title Slide (0:30)**
*[Look confident, smile, make eye contact]*

"Good [morning/afternoon], everyone. I'm Sumit Mahesh Khobragade from the Australian National University. Today I'll be presenting my research on 'Ego Motion Compensation in Event Cameras Using Forward Prediction.' This work addresses a fundamental challenge in event-based vision systems and introduces a novel causal approach for real-time motion compensation."

---

## **Slide 2: Event Cameras 101 (1:40)**
*[Point to the visual comparison]*

"Let me start by explaining what event cameras are and why they're revolutionary. Unlike traditional cameras that capture frames at fixed intervals, event cameras report brightness changes asynchronously with microsecond latency and high dynamic range."

*[Gesture to the formula]*

"Each event encodes four pieces of information: location (x,y), time (t), and polarity (p). The mathematical condition for event generation is simple: when brightness changes exceed a threshold C, an event is triggered."

*[Point to advantages]*

"The key advantages are clear: low latency, high dynamic range, and motion blur immunity. This makes them ideal for high-speed applications like autonomous vehicles and robotics. However, there's a significant challenge..."

---

## **Slide 3: Ego Motion Problem + Why It Matters (1:10)**
*[Use hand gestures to illustrate the problem]*

"The challenge is ego-motion. When the camera moves, it creates dense 'ego events' even in static scenes. Imagine spinning a camera - every pixel sees brightness changes due to motion, creating thousands of events per second."

*[Point to the impact points]*

"This creates three major problems: First, novel object motion gets buried in noise. Second, it wastes bandwidth and computational resources. Third, it degrades the performance of downstream algorithms like SLAM and object detection."

*[Transition gesture]*

"The goal is clear: suppress predictable ego-motion events while preserving true scene dynamics. Let me show you how traditional approaches handle this..."

---

## **Slide 4: Traditional Solutions (1:25)**
*[Point to each approach]*

"Traditional approaches use post-hoc motion compensation. The Image of Warped Events method warps events to a reference time using motion models, optimizing for maximum contrast. This works well for complex motions but has significant limitations."

*[Highlight limitations]*

"These methods require batch processing over time windows, which breaks the event camera's asynchronous nature. They also introduce latency - you have to wait for a window of events before processing. Most importantly, they don't leverage the event camera's microsecond timing advantage."

*[Pause for emphasis]*

"Our approach is fundamentally different. Instead of post-hoc compensation, we use proactive prediction..."

---

## **Slide 5: Our Approach: Forward Prediction (1:40)**
*[Point to the core concept]*

"Our approach is proactive cancellation of predictable ego-motion events. The core concept is elegant: for each incoming event, we predict its future location at time t + Δt and emit an inverse-polarity anti-event to cancel predictable motion."

*[Gesture the process]*

"This is causal, per-event processing with no batch windows. The key advantages are clear: real-time operation, low latency, and it preserves the event camera's asynchronous nature."

*[Point to the mathematical foundation]*

"The mathematical foundation is the rotation model: we rotate each event about the motion center by an angle θ = ω·Δt. This gives us the predicted location for cancellation."

*[Transition]*

"Let me show you exactly how this works in practice..."

---

## **Slide 6: Core Algorithm (2:10)**
*[Use the flow diagram visual]*

"The algorithm is straightforward and elegant. For each incoming event at time t, we follow four steps:"

*[Point to each step]*

"Step 1: Event arrives with coordinates (x,y), polarity p, and timestamp t. Step 2: We sample motion parameters - rotation center (cx,cy) and angular velocity ω - at exactly time t. Step 3: We apply the rotation transformation to get the predicted location (x',y'). Step 4: We emit an anti-event at time t + Δt with flipped polarity."

*[Point to the mathematical formulas]*

"The rotation formulas are standard: x' = cx + cos(θ)(x-cx) - sin(θ)(y-cy), where θ = ω·Δt. The anti-event has coordinates (x', y', 1-p, t+Δt)."

*[Emphasize key properties]*

"This is truly causal - we only use past and present motion information. It's per-event processing with no batching. And it maintains microsecond latency, preserving the event camera's key advantage."

---

## **Slide 7: Matching/Cancellation Policy (1:10)**
*[Point to the three constraints]*

"Now, how do we match predicted anti-events with real events? We use three constraints:"

*[Point to temporal gate]*

"First, the temporal gate: |t_j - (t_i + Δt)| ≤ ε_t. This ensures the predicted anti-event arrives at the correct future time. It's our primary constraint."

*[Point to spatial gate]*

"Second, the spatial gate: √[(x_j-x'_i)² + (y_j-y'_i)²] ≤ ε_xy. This accounts for prediction accuracy and sensor noise using Euclidean distance."

*[Point to polarity]*

"Third, polarity matching: p_j ≠ p_i. The anti-event must have opposite polarity for effective cancellation."

*[Point to algorithm]*

"The matching algorithm processes each real event individually, finds the best predicted match within both gates, and marks matched pairs as cancelled. Residual events are the unmatched ones."

---

## **Slide 8: Parameters That Affect Cancellation (1:10)**
*[Point to each parameter]*

"Three key parameters control cancellation performance:"

*[Point to temporal tolerance]*

"Temporal tolerance ε_t controls the time window for matching. Too small, and you get few matches. Too large, and you get false matches. The optimal range is 2-5ms for typical motion."

*[Point to spatial tolerance]*

"Spatial tolerance ε_xy controls spatial proximity. Too small, and you miss valid matches due to noise. Too large, and you match distant events incorrectly. The optimal range is 1-3 pixels for high-resolution sensors."

*[Point to prediction time step]*

"Prediction time step Δt is the forward prediction horizon. Too small, and you have insufficient motion compensation. Too large, and prediction becomes inaccurate. The optimal range is 2-5ms to match sensor latency."

*[Point to motion parameters]*

"Motion parameters like angular velocity ω and rotation center (cx,cy) also affect performance, as they determine prediction accuracy."

---

## **Slide 9: Results (2:00)**
*[Point to the main visual - per_pixel_images.png]*

"Now for the results. This visual demonstration shows the dramatic effectiveness of our approach."

*[Point to Real events]*

"The 'Real' panel shows raw event data with dense ego-motion events from the spinning disc."

*[Point to Predicted events]*

"The 'Predicted' panel shows our algorithm's generated anti-events, mirroring the ego-motion pattern."

*[Point to Combined]*

"The 'Combined' panel is the key result - dramatic reduction in event density where ego-motion was predicted. Notice how the static hand object is preserved while the spinning disc events are suppressed."

*[Point to quantitative results]*

"Quantitatively, we achieved up to 100% cancellation rate at Δt = 0ms, decreasing to about 32% at Δt = 20ms. The optimal range of 2-5ms provides 70-90% cancellation."

*[Point to statistical improvement]*

"Statistically, we reduced median event count per pixel from 1.0 to 0.3, and the 95th percentile from 2.0 to 1.0. ROI analysis shows 77% cancellation inside the disc region and 56% outside."

*[Emphasize consistency]*

"Most importantly, performance is highly consistent across different time windows, demonstrating the robustness of our approach."

---

## **Slide 10: Future Directions + Conclusion (1:30)**
*[Point to immediate next steps]*

"Looking forward, our immediate next steps include real-time implementation on FPGA or GPU for microsecond latency, testing with dynamic scenes like pendulums, and integrating IMU sensors for improved motion estimation."

*[Point to technical enhancements]*

"Technical enhancements include adaptive parameter tuning, multi-scale processing, and extending beyond pure rotation to complex 6DOF motion."

*[Point to applications]*

"The broader impact spans autonomous vehicles, robotics, AR/VR, and scientific imaging - anywhere high-speed motion compensation is needed."

*[Point to key contributions]*

"Our key contributions are clear: we developed the first causal, per-event forward prediction for ego-motion, achieved proven effectiveness with up to 100% cancellation rates, maintained real-time capability, and demonstrated selective cancellation that preserves novel object motion."

*[Final emphasis]*

"This work opens new possibilities for real-time event processing without sacrificing the event camera's fundamental advantages of low latency and high temporal resolution."

*[Pause, then conclude]*

"Thank you for your attention. I'm happy to take questions."

---

## **Timing Summary:**
- Slide 1: 0:30 (cumulative: 0:30)
- Slide 2: 1:40 (cumulative: 2:10)
- Slide 3: 1:10 (cumulative: 3:20)
- Slide 4: 1:25 (cumulative: 4:45)
- **Micro-pause: 0:10 (cumulative: 4:55)**
- Slide 5: 1:40 (cumulative: 6:35)
- Slide 6: 2:10 (cumulative: 8:45)
- Slide 7: 1:10 (cumulative: 9:55)
- **Micro-pause: 0:10 (cumulative: 10:05)**
- Slide 8: 1:10 (cumulative: 11:15)
- Slide 9: 2:00 (cumulative: 13:15)
- **Micro-pause: 0:10 (cumulative: 13:25)**
- Slide 10: 1:30 (cumulative: 14:55)
- **Buffer: 0:05 (total: 15:00)**

## **Speaking Tips:**
1. **Practice the transitions** between slides
2. **Use hand gestures** to emphasize key points
3. **Make eye contact** with different parts of the audience
4. **Pause briefly** at the micro-pause points for water/breather
5. **Adjust timing** based on audience engagement
6. **Have backup explanations** ready for complex concepts
7. **End with confidence** and invite questions

## **Backup Material:**
- Keep the detailed visualizations ready for Q&A
- Have specific numbers memorized for cancellation rates
- Prepare examples of applications if asked
- Know the mathematical formulas by heart










