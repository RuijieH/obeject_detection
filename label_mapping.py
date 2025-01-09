class Label:
    def __init__(self, name, trainId):
        self.name = name           # 类别名称
        self.trainId = trainId     # 原始的类别 ID
label2label = {
    1: Label('person', 1 ),
    2: Label('rider', 2 ),
    3: Label('car', 3 ),
    4: Label('truck', 4 ),
    5: Label('bus', 5 ),
    6: Label('on rails', 6 ),
    7: Label('motorcycle', 7 ),
    8: Label('bicycle', 8 ),
    9: Label('caravan', 9),
    10: Label('trailer', 10)
}
# 根据label2label生成name2label
name2label = {label.name: label for label in label2label.values()}



#cityscape 完整类别为，需要检测新的类别可以在label2label中添加，注意0是fasterrcnn pytroch实现的背景类，不要将新添加的类名设置为0
# classes = [
#     CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
#     CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
#     CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
#     CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
#     CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
#     CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
#     CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
#     CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
#     CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
#     CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
#     CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
#     CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
#     CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
#     CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
#     CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
#     CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
#     CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
#     CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
#     CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
#     CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
#     CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
#     CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
#     CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
#     CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
#     CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
#     CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
#     CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
#     CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
#     CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
#     CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
#     CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
#     CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
#     CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
#     CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
#     CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
# ]