import cv2
import numpy as np
import pickle
import random
import time

# ── load your trained model ───────────────────────────────────────────────────

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["K1"],d["K2"],d["K3"],d["K4"],d["W1"],d["W2"],d["W3"],d["W4"],d["b1"],d["b2"],d["b3"],d["b4"]

# ── forward pass (copied from forward.py) ────────────────────────────────────

def conv2D(input, kernel, stride=1, padding=0):
    batch_size, in_depth, in_height, in_width = input.shape
    out_channels, __, kernel_height, kernel_width = kernel.shape
    input_padded = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    out_height = (in_height + 2*padding - kernel_height) // stride + 1
    out_width  = (in_width  + 2*padding - kernel_width)  // stride + 1
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    for n in range(batch_size):
        for c in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    region = input_padded[n, :, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                    output[n, c, i, j] = np.sum(region * kernel[c])
    return output

def relu(x):
    return np.where(x > 0, x, 0.01 * x)

def max_pooling(input, pool_size, stride):
    batch_size, in_depth, in_height, in_width = input.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width  = (in_width  - pool_size) // stride + 1
    output = np.zeros((batch_size, in_depth, out_height, out_width))
    for n in range(batch_size):
        for c in range(in_depth):
            for i in range(out_height):
                for j in range(out_width):
                    region = input[n, :, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                    output[n, c, i, j] = np.max(region)
    return output

def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(img, K1, K2, K3, K4, W1, W2, W3, W4, b1, b2, b3, b4):
    pool1 = max_pooling(relu(conv2D(img,   K1, padding=1)), 2, 2)
    pool2 = max_pooling(relu(conv2D(pool1, K2, padding=1)), 2, 2)
    pool3 = max_pooling(relu(conv2D(pool2, K3, padding=1)), 2, 2)
    pool4 = max_pooling(relu(conv2D(pool3, K4, padding=1)), 2, 2)
    out   = relu(dense(flatten(pool4), W1, b1))
    out   = relu(dense(out, W2, b2))
    out   = relu(dense(out, W3, b3))
    probs = softmax(dense(out, W4, b4))
    return np.argmax(probs, axis=1)[0], probs[0]

def preprocess_frame(frame, target_size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resize first to match training preprocessing exactly
    resized = cv2.resize(gray, target_size)
    normalized = resized.astype('float32') / 255.0
    return normalized[np.newaxis, np.newaxis, :, :]

# ── game logic ────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["rock", "paper", "scissors"]
EMOJI        = {"rock": "✊", "paper": "✋", "scissors": "✌️"}
BEATS        = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

def get_winner(human, ai):
    if human == ai:
        return "draw"
    if BEATS[human] == ai:
        return "human"
    return "ai"

# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, thickness)
    cv2.ellipse(img, (x+r,   y+r),   (r,r), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x+w-r, y+r),   (r,r), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x+r,   y+h-r), (r,r), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x+w-r, y+h-r), (r,r), 0,   0, 90,  color, thickness)

def put_text(img, text, pos, scale, color, thickness=2, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, pos, font, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, font, scale, color,   thickness)

# ── main app ─────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    weights = load_model("model.pkl")
    print("Model loaded. Opening camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # game state
    scores       = {"human": 0, "ai": 0, "draw": 0}
    state        = "waiting"   # waiting | countdown | result
    countdown    = 3
    countdown_t  = 0
    result_t     = 0
    human_move   = None
    ai_move      = None
    winner       = None
    last_pred    = "rock"
    last_conf    = 0.0

    # ROI box for hand — right side of frame
    ROI_X, ROI_Y, ROI_W, ROI_H = 720, 80, 500, 500

    DARK   = (18,  18,  30)
    ACCENT = (80,  220, 130)
    RED    = (60,  80,  240)
    GOLD   = (30,  200, 255)
    WHITE  = (240, 240, 240)
    GRAY   = (120, 120, 140)

    print("\n  ROCK PAPER SCISSORS")
    print("  Place hand in the GREEN box")
    print("  Press SPACE to play a round")
    print("  Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]

        # ── live prediction from ROI ──────────────────────────────────────────
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            inp         = preprocess_frame(roi)
            pred_idx, probs = predict(inp, *weights)
            last_pred   = CLASS_NAMES[pred_idx]
            last_conf   = float(probs[pred_idx])

        # ── dark overlay canvas ───────────────────────────────────────────────
        canvas = frame.copy()
        overlay = np.zeros_like(canvas)
        overlay[:] = DARK
        canvas = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)


        # show what the model sees — helps debug
        debug_img = cv2.resize(
            cv2.cvtColor((inp[0,0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            (150, 150)
        )
        canvas[ROI_Y:ROI_Y+150, ROI_X-160:ROI_X-10] = debug_img
        put_text(canvas, "MODEL VIEW", (ROI_X-160, ROI_Y-5), 0.45, GRAY, 1)



        # ── ROI border ────────────────────────────────────────────────────────
        border_color = ACCENT if state == "waiting" else (GOLD if state == "countdown" else RED)
        cv2.rectangle(canvas, (ROI_X-3, ROI_Y-3), (ROI_X+ROI_W+3, ROI_Y+ROI_H+3), border_color, 3)
        put_text(canvas, "PLACE HAND HERE", (ROI_X, ROI_Y - 15), 0.65, border_color)

        # ── title ─────────────────────────────────────────────────────────────
        put_text(canvas, "ROCK  PAPER  SCISSORS", (30, 55), 1.1, ACCENT, 3)

        # ── score panel ───────────────────────────────────────────────────────
        draw_rounded_rect(canvas, 20, 80, 280, 150, 12, (30, 32, 45), -1)
        put_text(canvas, "YOU", (40, 120), 0.7, GRAY)
        put_text(canvas, str(scores["human"]), (40, 175), 2.2, WHITE, 4)
        put_text(canvas, "AI", (180, 120), 0.7, GRAY)
        put_text(canvas, str(scores["ai"]), (180, 175), 2.2, RED, 4)

        # ── live detection ────────────────────────────────────────────────────
        draw_rounded_rect(canvas, 20, 250, 280, 110, 12, (30, 32, 45), -1)
        put_text(canvas, "DETECTING", (40, 285), 0.6, GRAY)
        put_text(canvas, last_pred.upper(), (40, 335), 1.1, ACCENT, 3)
        conf_w = int(230 * last_conf)
        cv2.rectangle(canvas, (40, 345), (40 + conf_w, 355), ACCENT, -1)
        cv2.rectangle(canvas, (40, 345), (270,          355), GRAY,   1)
        put_text(canvas, f"{last_conf*100:.0f}%", (185, 360), 0.55, GRAY, 1)

        # ── instructions ──────────────────────────────────────────────────────
        draw_rounded_rect(canvas, 20, 380, 280, 100, 12, (30, 32, 45), -1)
        put_text(canvas, "SPACE  play round", (35, 410), 0.55, GRAY, 1)
        put_text(canvas, "R      reset score", (35, 438), 0.55, GRAY, 1)
        put_text(canvas, "Q      quit",        (35, 466), 0.55, GRAY, 1)
        
        now = time.time()

        # ── state machine ─────────────────────────────────────────────────────
        if state == "countdown":
            elapsed = now - countdown_t
            remaining = countdown - int(elapsed)
            if remaining > 0:
                num = str(remaining)
                put_text(canvas, num, (W//2 - 40, H//2 + 60), 5.0, GOLD, 8)
                put_text(canvas, "GET READY", (W//2 - 120, H//2 - 60), 1.2, WHITE, 3)
            else:
                # lock in the prediction
                human_move = last_pred
                ai_move    = random.choice(CLASS_NAMES)
                winner     = get_winner(human_move, ai_move)
                scores[winner] += 1
                state    = "result"
                result_t = now

        elif state == "result":
            elapsed = now - result_t

            # result panel
            draw_rounded_rect(canvas, W//2 - 280, H//2 - 140, 560, 300, 20, (22, 24, 38), -1)

            # winner text
            if winner == "human":
                w_text, w_color = "YOU WIN!", ACCENT
            elif winner == "ai":
                w_text, w_color = "AI WINS!", RED
            else:
                w_text, w_color = "DRAW!", GOLD

            put_text(canvas, w_text, (W//2 - 130, H//2 - 70), 2.0, w_color, 4)

            # moves
            put_text(canvas, f"You: {human_move.upper()}", (W//2 - 240, H//2 + 10), 0.9, WHITE, 2)
            put_text(canvas, f"AI:  {ai_move.upper()}",    (W//2 + 30,  H//2 + 10), 0.9, WHITE, 2)

            # prob bars
            put_text(canvas, "Confidence:", (W//2 - 240, H//2 + 55), 0.6, GRAY, 1)
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, [last_conf if CLASS_NAMES[i]==human_move else 0 for i in range(3)])):
                pass
            # show all 3 class probs
            for i, name in enumerate(CLASS_NAMES):
                bx = W//2 - 240 + i * 185
                put_text(canvas, name[:3].upper(), (bx, H//2 + 90), 0.55, GRAY, 1)

            put_text(canvas, "SPACE for next round", (W//2 - 170, H//2 + 130), 0.65, GRAY, 1)

            if elapsed > 5:
                state = "waiting"

        else:  # waiting
            put_text(canvas, "Press SPACE to play", (W//2 - 185, H - 40), 0.85, GRAY, 2)

        cv2.imshow("Rock Paper Scissors", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if state == "waiting":
                state       = "countdown"
                countdown_t = time.time()
            elif state == "result":
                state = "waiting"
        elif key == ord('r'):
            scores = {"human": 0, "ai": 0, "draw": 0}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()