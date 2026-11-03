
import cv2
import numpy as np
from pathlib import Path

OUTPUT = Path("dataset/forms")
OUTPUT.mkdir(parents=True, exist_ok=True)

# A4-like document canvas
WIDTH = 1000
HEIGHT = 1200

image = np.ones(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
) * 255

# Heading
cv2.putText(
    image,
    "DEVANAGARI HANDWRITTEN FORM",
    (180, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    2
)

cv2.putText(
    image,
    "Please fill all fields by hand",
    (300, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (80, 80, 80),
    1
)

# Field definitions: name, label, rectangle
fields = [
    ("name", "NAME", (100, 170, 800, 115)),
    ("date", "DATE", (100, 370, 800, 100)),
    ("address", "ADDRESS", (100, 570, 800, 180)),
    ("amount", "AMOUNT", (100, 850, 800, 115))
]

for field_name, label, (x, y, w, h) in fields:

    cv2.putText(
        image,
        label,
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        2
    )

# Footer
cv2.putText(
    image,
    "FormDigitizer AI - Handwritten Document Processing",
    (150, 1080),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (100, 100, 100),
    1
)

output_path = OUTPUT / "blank_form.png"

cv2.imwrite(str(output_path), image)

print("Form generated successfully!")
print("Saved:", output_path.resolve())

cv2.imshow("Blank Form", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
