import cv2
import os
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


if __name__ == "__main__":
    for line in open("MOT17/train/ImageSets/Main/trainval.txt", "r"):
        line = line.rstrip()
        img_path = os.path.join("MOT17/train/JPEGImages", line + ".jpg")
        anno_path = os.path.join("MOT17/train/Annotations", line + ".xml")
        img = cv2.imread(img_path)
        anno = ET.parse(anno_path).getroot()
        file_name = anno.find('filename').text.lower().strip()
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        for obj in anno.iter('object'):
            bbox = obj.find('bndbox')
            box = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                box.append(cur_pt)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow("MOT17", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

