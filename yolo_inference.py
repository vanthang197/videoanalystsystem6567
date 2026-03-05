from ultralytics import YOLO

model = YOLO('yolo26n.pt')

results = model.predict(
    "input_videos/TTDNvsDT01.mp4",
    save=True,
    vid_stride=5,
    device=0,
    vid_stride=3
    )
print(results[0])
print('=======================')
for box in results[0].boxes:
    print(box)