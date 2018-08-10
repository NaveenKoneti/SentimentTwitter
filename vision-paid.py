import numpy as np
import cv2
import math
import keras
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr,preprocess_image,resize_image
import os
import numpy as np
import time
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')



model_path = '/home/navgill/Downloads/resnet50_coco_best_v1.2.2.h5'
model = keras.models.load_model(model_path, custom_objects=custom_objects)

def distance_calculation(input_image):
    def detect_object_marker_width(image):
        CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',\
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', \
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',\
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',\
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',\
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',\
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',\
                   47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', \
                   54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',\
                   61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', \
                   68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',\
                   75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        

        
        given_image = image
        given_image = preprocess_image(given_image)
        given_image, scale = resize_image(given_image)
        
        
        (h, w) = given_image.shape[:2]
        

        _, _, detections = model.predict_on_batch(np.expand_dims(given_image, axis=0))
        

        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]


        detections[0, :, :4] /= scale
        label_and_width = []
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score > 0.499:
                
                b = detections[0, idx, :4].astype(int)
                width = b[2] - b[0]   
                
                label_name = CLASSES[label]
                label_and_width = label_and_width + [(label_name,width)]
                #print("label: ", label_name)
                #print("width: ", width)         
            
        return label_and_width
 
    def distance_to_camera(knownWidth, focalLength, perWidth):
        
        return (knownWidth * focalLength) / perWidth



    def reference_data(Object,Object_path,Object_distance,Object_width):
        

        Object = Object
        KNOWN_DISTANCE = Object_distance
        KNOWN_WIDTH = Object_width
        reference = Object_path
        image = read_image_bgr(reference)
        #image, scale = resize_image(image)
        label_and_widths = detect_object_marker_width(image)
        #print(label_and_widths)
        req_label_and_width = [item for item in label_and_widths if item[0] == Object]
        
        req_label_and_width = max(req_label_and_width, key = lambda t: t[1])
        width = req_label_and_width[1]
        focalLength = (width * KNOWN_DISTANCE) / KNOWN_WIDTH
        #print("reference object width", str(width)+" "+ str(Object) )
        #print("##################################################")
        return Object,KNOWN_WIDTH,focalLength

    
    #baseballbat_data = reference_data("baseball bat","./images/Reference_Images/baseballbat.jpg",1.5, 0.12)
    apple_data = reference_data("apple","./images/Reference_Images/apple.jpg",1.5, 0.05)
    backpack_data = reference_data("backpack","./images/Reference_Images/backpack.jpg",2.0, 0.2)
    banana_data = reference_data("banana","./images/Reference_Images/banana.jpg",1.5, 0.07)
    #baseballbat_data = reference_data("baseball bat","./images/Reference_Images/baseballbat.jpg",1.5, 0.12)
    bench_data = reference_data("bench","./images/Reference_Images/bench.jpg",2.0, 2.0)
    bicycle_data = reference_data("bicycle","./images/Reference_Images/bicycle.jpg",5.0,1.0)
    book_data = reference_data("book","./images/Reference_Images/book.jpg",0.7, 0.1)
    bottle_data = reference_data("bottle","./images/Reference_Images/bottle.jpg",2.0,0.1)
    bowl_data = reference_data("bowl","./images/Reference_Images/bowl.jpg",1.0,0.2)
    broccoli_data = reference_data("broccoli","./images/Reference_Images/broccoli.jpg",0.7, 0.1)
    bus_data = reference_data("bus","./images/Reference_Images/bus.jpg",40.0, 12.0)
    cake_data = reference_data("cake","./images/Reference_Images/cake.jpg",0.8, 0.3)
    car_data = reference_data("car","./images/Reference_Images/car.jpg",10.0,1.74)
    carrot_data = reference_data("carrot","./images/Reference_Images/carrot.jpg",0.7, 0.05)
    cellphone_data = reference_data("cell phone","./images/Reference_Images/cellphone.jpg",0.7, 0.07)
    chair_data = reference_data("chair","./images/Reference_Images/chair.jpg",3.0,1.0)
    clock_data = reference_data("clock","./images/Reference_Images/clock.jpg",1.5, 0.3)
    couch_data = reference_data("couch","./images/Reference_Images/couch.jpg",2.0, 2.0)
    cow_data = reference_data("cow","./images/Reference_Images/cow.jpg",6.0, 2.0)
    cup_data = reference_data("cup","./images/Reference_Images/cup.jpg",2.0,0.1)
    diningtable_data = reference_data("dining table","./images/Reference_Images/diningtable.jpg",4.0,2.0)
    dog_data = reference_data("dog","./images/Reference_Images/dog.jpg",5.0,1.0)
    donut_data = reference_data("donut","./images/Reference_Images/donut.jpg",0.5, 0.1)
    firehydrant_data = reference_data("fire hydrant","./images/Reference_Images/firehydrant.jpg",5.0, 1.0)
    hairdrier_data = reference_data("hair drier","./images/Reference_Images/hairdrier.jpg",0.7, 0.2)
    keyboard_data = reference_data("keyboard","./images/Reference_Images/keyboard.jpg",0.5,0.4)
    knife_data = reference_data("knife","./images/Reference_Images/knife.jpg",1.0, 0.15)
    laptop_data = reference_data("laptop","./images/Reference_Images/laptop.jpg",1.0,0.3)
    microwave_data = reference_data("microwave","./images/Reference_Images/microwave.jpg",2.0, 1.0)
    motorcycle_data = reference_data("motorcycle","./images/Reference_Images/motorcycle.jpg",3.0,1.0)
    orange_data = reference_data("orange","./images/Reference_Images/orange.jpg",1.5, 0.1)
    person_data = reference_data("person","./images/Reference_Images/person.jpg",7.0,0.8)
    pottedplant_data = reference_data("potted plant","./images/Reference_Images/pottedplant.jpg",2.0,0.5)
    remote_data = reference_data("remote","./images/Reference_Images/remote.jpg",1.0,0.1)
    sandwich_data = reference_data("sandwich","./images/Reference_Images/sandwitch.jpg",1.0,0.2)
    scissors_data = reference_data("scissors","./images/Reference_Images/scissors.jpg",1.5, 0.4)
    skateboard_data = reference_data("skateboard","./images/Reference_Images/skateboard.jpg",2.0, 1.0)
    sportsball_data = reference_data("sports ball","./images/Reference_Images/sportsball.jpg",2.0, 0.3)
    suitcase_data = reference_data("suitcase","./images/Reference_Images/suitcase.jpg",2.0, 1.0)
    teddybear_data = reference_data("teddy bear","./images/Reference_Images/teddybear.jpg",1.0, 0.5)
    tennisracket_data = reference_data("tennis racket","./images/Reference_Images/tennisracket.jpg",2.0, 0.5)
    tie_data = reference_data("tie","./images/Reference_Images/tie2.jpg",2.0, 0.15)
    toaster_data = reference_data("toaster","./images/Reference_Images/toaster.jpg",1.0, 0.2)
    toilet_data = reference_data("toilet","./images/Reference_Images/toilet.jpg",1.5, 1.0)
    trafficlight_data = reference_data("traffic light","./images/Reference_Images/trafficlight.jpg",20.0,1.0)
    truck_data = reference_data("truck","./images/Reference_Images/truck.jpg",15.0,4.0)
    tv_data = reference_data("tv","./images/Reference_Images/tv.jpg",4.0,1.2)
    umbrella_data = reference_data("umbrella","./images/Reference_Images/umbrella.jpg",2.0,1.5)
    vase_data = reference_data("vase","./images/Reference_Images/vase.jpg",2.0,0.1)
    wineglass_data = reference_data("wine glass","./images/Reference_Images/wineglass.jpg",1.0,0.2)
    
    
    

    IMAGE_PATHS = [input_image]
    for imagePath in IMAGE_PATHS:
        CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',\
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', \
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',\
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',\
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',\
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',\
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',\
                   47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', \
                   54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',\
                   61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', \
                   68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',\
                   75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        


        image = read_image_bgr(imagePath)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        (h, w) = image.shape[:2]
        

        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
        

        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
        
        
        #print(predicted_labels)

        detections[0, :, :4] /= scale

        image_centreX,image_centreY = (w/2),(h/2)

        all_objects = []
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score > 0.499:
                b = detections[0, idx, :4].astype(int)
                width = b[2]-b[0]   
                #print("idx:",idx)
                #print("Actual Object data is: ",str(width)+" " + str(label))  

                def Objects_data(object_data):
                    object_yards = distance_to_camera(object_data[1], object_data[2], width)
                    object_centreX,object_centreY = ((b[0]+b[2])/2),((b[1]+b[3])/2)
                    object_distance_from_centre = math.sqrt((object_centreX-image_centreX)**2 + (object_centreY-image_centreY)**2)
                    return object_data[0],object_distance_from_centre,object_yards
                                   
                if(CLASSES[label] == "apple"):
                    all_objects = all_objects + [(Objects_data(apple_data)[0],Objects_data(apple_data)[1],Objects_data(apple_data)[2])]
                if(CLASSES[label] == "backpack"):
                    all_objects = all_objects + [(Objects_data(backpack_data)[0],Objects_data(backpack_data)[1],Objects_data(backpack_data)[2])]
                if(CLASSES[label] == "banana"):
                    all_objects = all_objects + [(Objects_data(banana_data)[0],Objects_data(banana_data)[1],Objects_data(banana_data)[2])]
                #if(CLASSES[label] == "baseball bat"):
                #    all_objects = all_objects + [(Objects_data(baseballbat_data)[0],Objects_data(baseballbat_data)[1],\
                #                                  Objects_data(baseballbat_data)[2])]
                if(CLASSES[label] == "bench"):
                    all_objects = all_objects + [(Objects_data(bench_data)[0],Objects_data(bench_data)[1],Objects_data(bench_data)[2])]
                if(CLASSES[label] == "bicycle"):
                    all_objects = all_objects + [(Objects_data(bicycle_data)[0],Objects_data(bicycle_data)[1]\
                                                ,Objects_data(bicycle_data)[2])]
                if(CLASSES[label] == "book"):
                    all_objects = all_objects + [(Objects_data(book_data)[0],Objects_data(book_data)[1],Objects_data(book_data)[2])]
                if(CLASSES[label] == "bottle"):
                    all_objects = all_objects + [(Objects_data(bottle_data)[0],Objects_data(bottle_data)[1],Objects_data(bottle_data)[2])]
                if(CLASSES[label] == "bowl"):
                    all_objects = all_objects + [(Objects_data(bowl_data)[0],Objects_data(bowl_data)[1],Objects_data(bowl_data)[2])]
                if(CLASSES[label] == "broccoli"):
                    all_objects = all_objects + [(Objects_data(broccoli_data)[0],Objects_data(broccoli_data)[1],\
                                                  Objects_data(broccoli_data)[2])]
                if(CLASSES[label] == "bus"):
                    all_objects = all_objects + [(Objects_data(bus_data)[0],Objects_data(bus_data)[1],Objects_data(bus_data)[2])]
                if(CLASSES[label] == "cake"):
                    all_objects = all_objects + [(Objects_data(cake_data)[0],Objects_data(cake_data)[1],Objects_data(cake_data)[2])]
                if(CLASSES[label] == "car"):
                    all_objects = all_objects + [(Objects_data(car_data)[0],Objects_data(car_data)[1],Objects_data(car_data)[2])]
                if(CLASSES[label] == "carrot"):
                    all_objects = all_objects + [(Objects_data(carrot_data)[0],Objects_data(carrot_data)[1],Objects_data(carrot_data)[2])]
                if(CLASSES[label] == "cell phone"):
                    all_objects = all_objects + [(Objects_data(cellphone_data)[0],Objects_data(cellphone_data)[1],\
                                                  Objects_data(cellphone_data)[2])]
                if(CLASSES[label] == "chair"):
                    all_objects = all_objects + [(Objects_data(chair_data)[0],Objects_data(chair_data)[1],Objects_data(chair_data)[2])]
                if(CLASSES[label] == "clock"):
                    all_objects = all_objects + [(Objects_data(clock_data)[0],Objects_data(clock_data)[1],Objects_data(clock_data)[2])]
                if(CLASSES[label] == "couch"):
                    all_objects = all_objects + [(Objects_data(couch_data)[0],Objects_data(couch_data)[1],Objects_data(couch_data)[2])]
                if(CLASSES[label] == "cow"):
                    all_objects = all_objects + [(Objects_data(cow_data)[0],Objects_data(cow_data)[1],Objects_data(cow_data)[2])]
                if(CLASSES[label] == "cup"):
                    all_objects = all_objects + [(Objects_data(cup_data)[0],Objects_data(cup_data)[1],Objects_data(cup_data)[2])]
                if(CLASSES[label] == "dining table"):
                    all_objects = all_objects + [(Objects_data(diningtable_data)[0],Objects_data(diningtable_data)[1],\
                                                  Objects_data(diningtable_data)[2])]
                if(CLASSES[label] == "dog"):
                    all_objects = all_objects + [(Objects_data(dog_data)[0],Objects_data(dog_data)[1],Objects_data(dog_data)[2])]
                if(CLASSES[label] == "donut"):
                    all_objects = all_objects + [(Objects_data(donut_data)[0],Objects_data(donut_data)[1],Objects_data(donut_data)[2])]
                if(CLASSES[label] == "fire hydrant"):
                    all_objects = all_objects + [(Objects_data(firehydrant_data)[0],Objects_data(firehydrant_data)[1],\
                                                  Objects_data(firehydrant_data)[2])]
                if(CLASSES[label] == "hair drier"):
                    all_objects = all_objects + [(Objects_data(hairdrier_data)[0],Objects_data(hairdrier_data)[1],\
                                                  Objects_data(hairdrier_data)[2])]
                if(CLASSES[label] == "keyboard"):
                    all_objects = all_objects + [(Objects_data(keyboard_data)[0],Objects_data(keyboard_data)[1],\
                                                  Objects_data(keyboard_data)[2])]
                if(CLASSES[label] == "knife"):
                    all_objects = all_objects + [(Objects_data(knife_data)[0],Objects_data(knife_data)[1],Objects_data(knife_data)[2])]
                if(CLASSES[label] == "laptop"):
                    all_objects = all_objects + [(Objects_data(laptop_data)[0],Objects_data(laptop_data)[1],Objects_data(laptop_data)[2])]
                if(CLASSES[label] == "microwave"):
                    all_objects = all_objects + [(Objects_data(microwave_data)[0],Objects_data(microwave_data)[1],\
                                                  Objects_data(microwave_data)[2])]
                if(CLASSES[label] == "motorcycle"):
                    all_objects = all_objects + [(Objects_data(motorcycle_data)[0],Objects_data(motorcycle_data)[1]\
                                                ,Objects_data(motorcycle_data)[2])]
                if(CLASSES[label] == "orange"):
                    all_objects = all_objects + [(Objects_data(orange_data)[0],Objects_data(orange_data)[1],Objects_data(orange_data)[2])]
                if(CLASSES[label] == "person"):
                    all_objects = all_objects + [(Objects_data(person_data)[0],Objects_data(person_data)[1],Objects_data(person_data)[2])]
                if(CLASSES[label] == "potted plant"):
                    all_objects = all_objects + [(Objects_data(pottedplant_data)[0],Objects_data(pottedplant_data)[1]\
                                                 ,Objects_data(pottedplant_data)[2])]
                if(CLASSES[label] == "remote"):
                    all_objects = all_objects + [(Objects_data(remote_data)[0],Objects_data(remote_data)[1],Objects_data(remote_data)[2])]
                if(CLASSES[label] == "sandwich"):
                    all_objects = all_objects + [(Objects_data(sandwich_data)[0],Objects_data(sandwich_data)[1],Objects_data(sandwich_data)[2])]
                if(CLASSES[label] == "scissors"):
                    all_objects = all_objects + [(Objects_data(scissors_data)[0],Objects_data(scissors_data)[1],\
                                                   Objects_data(scissors_data)[2])]
                if(CLASSES[label] == "skateboard"):
                    all_objects = all_objects + [(Objects_data(skateboard_data)[0],Objects_data(skateboard_data)[1],\
                                                  Objects_data(skateboard_data)[2])]
                if(CLASSES[label] == "sports ball"):
                    all_objects = all_objects + [(Objects_data(sportsball_data)[0],Objects_data(sportsball_data)[1],\
                                                  Objects_data(sportsball_data)[2])]
                if(CLASSES[label] == "suitcase"):
                    all_objects = all_objects + [(Objects_data(suitcase_data)[0],Objects_data(suitcase_data)[1],Objects_data(suitcase_data)[2])]
                if(CLASSES[label] == "teddybear"):
                    all_objects = all_objects + [(Objects_data(teddybear_data)[0],Objects_data(teddybear_data)[1],\
                                                  Objects_data(teddybear_data)[2])]
                if(CLASSES[label] == "tennis racket"):
                    all_objects = all_objects + [(Objects_data(tennisracket_data)[0],Objects_data(tennisracket_data)[1],\
                                                  Objects_data(tennisracket_data)[2])]
                if(CLASSES[label] == "tie"):
                    all_objects = all_objects + [(Objects_data(tie_data)[0],Objects_data(tie_data)[1],Objects_data(tie_data)[2])]
                if(CLASSES[label] == "toaster"):
                    all_objects = all_objects + [(Objects_data(toaster_data)[0],Objects_data(toaster_data)[1],Objects_data(toaster_data)[2])]
                if(CLASSES[label] == "toilet"):
                    all_objects = all_objects + [(Objects_data(toilet_data)[0],Objects_data(toilet_data)[1],Objects_data(toilet_data)[2])]
                if(CLASSES[label] == "traffic light"):
                    all_objects = all_objects + [(Objects_data(trafficlight_data)[0],Objects_data(trafficlight_data)[1]\
                                                ,Objects_data(trafficlight_data)[2])]
                if(CLASSES[label] == "truck"):
                    all_objects = all_objects + [(Objects_data(truck_data)[0],Objects_data(truck_data)[1],Objects_data(truck_data)[2])]
                if(CLASSES[label] == "tv"):
                    all_objects = all_objects + [(Objects_data(tv_data)[0],Objects_data(tv_data)[1],Objects_data(tv_data)[2])]
                if(CLASSES[label] == "umbrella"):
                    all_objects = all_objects + [(Objects_data(umbrella_data)[0],Objects_data(umbrella_data)[1],Objects_data(umbrella_data)[2])]
                if(CLASSES[label] == "vase"):
                    all_objects = all_objects + [(Objects_data(vase_data)[0],Objects_data(vase_data)[1],Objects_data(vase_data)[2])]
                
                
                if(CLASSES[label] == "wine glass"):
                    all_objects = all_objects + [(Objects_data(wineglass_data)[0],Objects_data(wineglass_data)[1],\
                                                  Objects_data(wineglass_data)[2])]
                
                
                try:
                    centre_object = min(all_objects, key = lambda t: t[1])
                except ValueError:
                    continue
        required_object = centre_object[0]        
        required_object_distance = centre_object[2]
        print("The object is approximately at a distance: ", round(required_object_distance,2), " yards")

###VISION Engine
def visionengine(input_image):
        #print('Processing...  ')
        import time
        import cloudsight
        import PIL
        from PIL import Image

        resizestart = time.time()

        basewidth = 400


        auth = cloudsight.OAuth('hOYp5eBuXW7CjONn3wsG6w', 'Oe3vUUIOHVIiBNfZSXynbA')
        api = cloudsight.API(auth)
        
	
        #image = 'img2'
        original_image_start = time.time()

        
        

        #auth = cloudsight.OAuth('hOYp5eBuXW7CjONn3wsG6w', 'Oe3vUUIOHVIiBNfZSXynbA')
	#api = cloudsight.API(auth)
        
        img = Image.open(input_image)
        width, height = img.size
        new_width = 0.5*width
        new_height = 0.7 * height
        left = int((width - new_width)/2)
        top = int((height - new_height)/2)
        right = int((width + new_width)/2)
        bottom = int((height + new_height)/2)
        img = img.crop((left,top,right,bottom))
        img.save('newimage.jpg')

        def imageResponse(image_path):
            cropped_image_start = time.time()
            with open(image_path, 'rb') as f:
                response = api.image_request(f, 'your-file.jpg', {
                    'image_request[locale]': 'en-US',
                     })

	    #response = api.remote_image_request('http://www.example.com/image.jpg', {
		    #'image_request[locale]': 'en-US',
                                                     #})
            while 1:
                status = api.image_response(response['token'])
                if status['status'] != 	cloudsight.STATUS_NOT_COMPLETED:
                    # Done!
                    #pass
                    break
                else:
                    time.sleep(0.01)

	        # Please also test this method if it returns results faster than method above.
	        # Cloudsight documentation says to check for results every second and that average speed of results is 4 to 6 seconds.
	        #status = api.wait(response['token'], timeout=10)

            processing_time = round(time.time() - cropped_image_start,2)

            results = status['name']

            return results,processing_time

	    #print('I think this shows:  ' +'%s'% results)




        try:
            try:
                result = imageResponse("newimage.jpg")
                TimeTaken = result[1]
                #print('Total Processing Time: ' +'%s'% TimeTaken +' seconds... ')
                print('I think this shows:  ' +'%s'% result[0])
            except KeyError:
                result = imageResponse('imagesNEW/'+image + '.jpg')
                TimeTaken = result[1]
                print('Total Processing Time: ' +'%s'% TimeTaken +' seconds... ')
                print('I think this shows:  ' +'%s'% result[0])
        except KeyError:
            print('I am unable to read the situation')
        
                
                
if __name__ == '__main__':
	while 1:
                #visioncam()
                image_path = "./images/Reference_Images/wineglass.jpg"
                start = time.time()
                distance_calculation(image_path)
                visionengine(image_path)
                end = time.time()
                print(end-start)
                break
