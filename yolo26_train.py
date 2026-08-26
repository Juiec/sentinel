import ultralytics


if __name__ == '__main__':
    model = ultralytics.YOLO("models\yolo26n.pt")
    results = model.train(
        data=r"C:\yolo-env\dataset\Backpack\data.yaml",  # dataset.yaml path
        device=0,             # use GPU 0 (or "cpu" for CPU)
        epochs=100,           # number of training epochs
        imgsz=640,            # image size (pixels)
        batch=16,             # batch size
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        degrees=15.0,
        erasing=0.1,
        #------------#        #Explicitly disabled settings
        flipud=0.0,
        bgr=0.0,
        perspective=0.0,
        shear=0.0,
        name="backpack"     # subfolder name -> output lands in runs/backpack/
    )