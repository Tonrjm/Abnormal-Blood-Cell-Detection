from ultralytics import YOLO

def main():
    # 1. โหลดโมเดลตั้งต้น
    model = YOLO("yolov8n.pt") 

    # 2. สั่งเทรนภายในฟังก์ชันหลัก
    results = model.train(
        data="BloodCellDetect.v3i.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        device=0,          # สั่งรันบน RTX 3050
        batch=32,          # ส่งรูปทีละ 32 รูป
        workers=4,         # ใช้ CPU ช่วยโหลดภาพขนาน 4 threads
    )

if __name__ == '__main__':
    main()  # ป้องกันปัญหาระบบ Multiprocessing บน Windows