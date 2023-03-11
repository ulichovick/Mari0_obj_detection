from mari0_record_test_modified import *

path = r'/home/ulichovick/Imágenes/labels_test.png'
INPUT_WIDTH = 800
INPUT_HEIGHT = 448
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
    
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
    
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)


def run():
    classes = ['Coin_Block', 'Destroyable_Tile', 'Flag_Pole', 'Goomba', 'Koopa', 'Little_Mario', 'Mario dead', 'Pipe', 'Pipe_Tile', 'Tile', 'Win_Flag', 'floor']
    # Load class names.
    modelWeights = "best_reworked_dataset_v4.onnx"
    net = cv2.dnn.readNet(modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    img = cv2.imread(path) #x, y, w, h 
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # Give the weight files to the model and load the network using them.
    # Process image.
    detections = pre_process(frame, net)
    img, boxes, class_ids, indices = post_process(frame.copy(), detections)
    t, _ = net.getPerfProfile()
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(boxes)
    print(indices)
    for i in indices:
        print(str(classes[class_ids[i]]) +"\n"+ str(i))

    print(class_ids)


if __name__ == '__main__':
    run()