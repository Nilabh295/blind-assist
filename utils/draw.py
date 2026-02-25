import cv2
import numpy as np

# ── Colours (BGR) ─────────────────────────────────────────────────────────
C = {
    "bg":      ( 15,  17,  26),
    "panel":   ( 22,  26,  40),
    "card":    ( 32,  38,  56),
    "border":  ( 55,  62,  88),
    "danger":  ( 45,  45, 220),
    "warn":    ( 30, 140, 255),
    "safe":    ( 80, 200,  80),
    "blue":    (220, 160,  50),
    "white":   (235, 235, 240),
    "gray":    (140, 145, 165),
    "dim":     ( 75,  80, 100),
    "flash":   ( 30,  30, 180),
}
ZCOL = {"danger": C["danger"], "warn": C["warn"], "safe": C["safe"]}
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTD = cv2.FONT_HERSHEY_DUPLEX

# ══════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _fill(img, x1, y1, x2, y2, col, alpha=1.0):
    if alpha < 1.0:
        ov = img.copy()
        cv2.rectangle(ov, (x1,y1),(x2,y2), col, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    else:
        cv2.rectangle(img, (x1,y1),(x2,y2), col, -1)

def _txt(img, t, x, y, col=None, sc=0.5, th=1, font=FONT):
    cv2.putText(img, t, (x,y), font, sc, col or C["white"], th, cv2.LINE_AA)

def _badge(img, t, x, y, bg, fg=None):
    fg = fg or C["white"]
    (tw,th),_ = cv2.getTextSize(t, FONT, 0.42, 1)
    cv2.rectangle(img, (x, y-th-4),(x+tw+8, y+4), bg, -1)
    cv2.putText(img, t, (x+4,y), FONT, 0.42, fg, 1, cv2.LINE_AA)

def _corner_box(img, x1,y1,x2,y2, col, thick=2):
    """Draw bounding box with stylish corner markers."""
    cv2.rectangle(img,(x1,y1),(x2,y2), col, thick)
    L = 16
    for sx,sy,ex,ey in [(x1,y1,x1+L,y1),(x2-L,y1,x2,y1),
                         (x1,y2,x1+L,y2),(x2-L,y2,x2,y2)]:
        cv2.line(img,(sx,sy),(ex,ey),col,3)
    for sx,sy,ex,ey in [(x1,y1,x1,y1+L),(x2,y1,x2,y1+L),
                         (x1,y2,x1,y2-L),(x2,y2,x2,y2-L)]:
        cv2.line(img,(sx,sy),(ex,ey),col,3)
