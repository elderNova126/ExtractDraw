# -*- coding: utf-8 -*-
import numpy as np
from io import BytesIO
import cv2
import fitz
import pytesseract
from pytesseract import Output
import os
import requests
from dotenv import load_dotenv
import base64
import rpack
from skimage.metrics import structural_similarity as ssim
from itertools import cycle

load_dotenv()

tesseract_Path = os.getenv("TESSERACT_PATH")
api_keys = os.getenv("OPENAI_KEY", "").split(",")
api_key_cycle = cycle(api_keys)  # Creates an infinite loop over API keys
pytesseract.pytesseract.tesseract_cmd = tesseract_Path

def subset(set, lim, loc):
        '''
        set: one or multi list or array, lim: size, loc:location(small, medi, large)
        This function reconstructs set according to size of lim in location of loc.
        '''
        cnt, len_set = 0, len(set)        
        v_coor_y1, index_ = [], []
        pop = []
        for i in range(len_set):
            if i < len_set-1:
                try:
                    condition = set[i+1][0] - set[i][0]
                except:
                    condition = set[i+1] - set[i]
                if condition < lim:
                    cnt = cnt + 1
                    pop.append(set[i])
                else:
                    cnt = cnt + 1
                    pop.append(set[i])
                    pop = np.asarray(pop)
                    try:
                        if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                        elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                        else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    except:
                        if loc == "small": v_coor_y1.append(min(pop))
                        elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                        else: v_coor_y1.append(max(pop))  
                    index_.append(cnt)
                    cnt = 0
                    pop = []
            else:
                cnt += 1
                pop.append(set[i])
                pop = np.asarray(pop)
                try:
                    if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                    else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                except:
                    if loc == "small": v_coor_y1.append(min(pop))
                    elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                    else: v_coor_y1.append(max(pop))                    
                index_.append(cnt)

        return v_coor_y1, index_ 

def split_pages(digit_doc, past_k, next_k):
    '''
    1. Splits the input pdf into pages
    2. Writes a temporary image for each page to a byte buffer
    3. Loads the image as a numpy array using cv2.imread()
    4. Appends the page image/array to self.pages

    Notes:
    PyMuPDF's get_pixmap() has a default output of 96dpi, while the desired
    resolution is 300dpi, hence the zoom factor of 300/96 = 3.125 ~ 3.
    '''
    print("Splitting PDF into pages")
    pages = []
    try:
        zoom_factor = 3
        for i in range(past_k, next_k):
            # Load page and get pixmap
            page = digit_doc.load_page(i)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))

            # Initialize bytes buffer and write PNG image to buffer
            buffer = BytesIO()
            buffer.write(pixmap.tobytes())
            buffer.seek(0)

            # Load image from buffer as array, append to pages, close buffer
            img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            page_img = cv2.imdecode(img_array, 1)
            pages.append(page_img)
            buffer.close()
    except:
        pass
    if len(pages) == 0:
        val = "01"
    else:
        val = pages
    return val

def approximate(li, limit):
    pre_l = li[0]
    new_li = []
    for l in li:
        if abs(l - pre_l) < limit:
            l = pre_l
        else:
            pre_l = l
        new_li.append(l)
    return new_li

def line_remove(image):
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    return result

def border_set(img_, coor, tk, color):
    '''
    coor: [x0, x1, y0, y1] - this denotes border locations.
    tk: border thickness, color: border color.
    '''
    img = img_.copy()
    if coor[0] != None:
        img[:, coor[0]:coor[0]+tk] = color # left vertical
    if coor[1] != None:
        img[:, coor[1]-tk:coor[1]] = color # right vertical
    if coor[2] != None:                    
        img[coor[2]:coor[2]+tk,:] = color # top horizontal
    if coor[3] != None:
        img[coor[3]-tk:coor[3],:] = color # bottom horizontal          
    return img  

def is_near_duplicate(existing_points, new_point, threshold=5):
    x_new, y_new, r_new = new_point
    
    for x, y, r in existing_points:
        if abs(x - x_new) <= threshold and abs(y - y_new) <= threshold:
            return True
    
    return False

def get_unique_points(points, threshold=5):
    unique_points = []
    
    for point in points:
        if not is_near_duplicate(unique_points, point, threshold):
            unique_points.append(point)
    
    return unique_points

def compare_images(image1, image2):
    # Resize image2 to match the dimensions of image1
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convert images to grayscale (SSIM works better with grayscale images)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    similarity_index, _ = ssim(gray1, gray2, full=True)   

    return similarity_index 

def find_small_circles(image, min_radius=10, max_radius=25, max_region_size=600):
    height, width = image.shape[:2]
    circles = []
    ballooned_number_s = 40
    
    # Define step sizes ensuring regions are within max_region_size
    x_steps = list(range(0, width, max_region_size)) + [width]
    y_steps = list(range(0, height, max_region_size)) + [height]
    for i in range(len(x_steps) - 1):
        for j in range(len(y_steps) - 1):
            x_start, x_end = x_steps[i], x_steps[i + 1] + ballooned_number_s
            y_start, y_end = y_steps[j], y_steps[j + 1] + ballooned_number_s
            
            region = image[y_start:y_end, x_start:x_end]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                # Find contours
            
            contours, _ = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(region, contours, -1, (0, 255, 0), 2) 
            
            # Iterate through contours and filter based on width and height
            
            for contour in contours:
                area = cv2.contourArea(contour)
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                err=abs((area-np.pi*radius*radius)/(np.pi*radius*radius))
                if err< 0.15 and 15<radius<24:
                    cv2.drawContours(region, contour, -1, (255, 255, 255), 5)
                    # cv2.circle(region, (int(cx), int(cy)), int(radius)+1, (255, 255, 255), 4) 
                    global_x = int(cx + x_start)
                    global_y = int(cy + y_start)
                    circles.append((global_x, global_y, int(radius)))
    
    return get_unique_points(circles)

def getting_textdata(img, conf):
    '''
    img: soucr image to process.
    conf: tesseract conf (--psm xx)
    zoom_fac: image resize factor.
    split_val: factor to consider for coordinate of texts when image is splited into two parts
    '''
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=conf)
    text_ori = d['text']
    left_coor, top_coor, wid, hei, conf = d['left'], d['top'], d['width'], d['height'], d['conf']        
    ### removing None element from text ###
    text, left, top, w, h, accu, xc, yc= [], [], [], [], [], [], [], []
    for cnt, te in enumerate(text_ori):
        if te.strip() != '' and wid[cnt] > 10 and hei[cnt] > 10:
            text.append(te)
            left.append(int((left_coor[cnt])))
            top.append(int(top_coor[cnt]))
            w.append(int(wid[cnt]))
            h.append(int(hei[cnt]))
            accu.append(conf[cnt])    
            xc.append(int((left_coor[cnt]+wid[cnt]/2)))
            yc.append(int((top_coor[cnt]+hei[cnt]/2)))
    return text, left, top, w, h, accu, xc, yc

def find_balloon_numbers(img, circles, IsDraw=False):
    detected_data = []
    H, W = img.shape[0:2]
    padding_size = 3
    intersect_img = cv2.imread('config/intersect.jpg')
    for i, (x, y, r) in enumerate(circles):

        x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)
        roi = img[y1+3:y2-3, x1+3:x2-3]
        
        new_roi = cv2.copyMakeBorder(roi, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        if compare_images(intersect_img, new_roi)>0.45:
            continue
        # cv2.circle(roi, (x, y), r, (255, 255, 255), thickness=4)  # Thin boundary removal
        # roi_cpy = roi.copy()
        # cv2.imwrite(f"img_{i}.jpg", roi_cpy)
        roi_gray = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(roi_thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789')
        text = text.strip()
        if text.isdigit():  # Ensure valid number detection
            detected_data.append({int(text):[x, y]})
    if IsDraw:
        imgCpy = img.copy()
        for (x, y, r) in circles:
            cv2.circle(imgCpy, (x, y), r, (0, 255, 0), 2)

        cv2.imwrite("upload/circled.jpg", imgCpy)
    return detected_data

def extract_and_place_subregions(image, rects, gap=5):
    subregions = [image[y0:y1, x0:x1] for (x0, y0), (x1, y1) in rects]
    max_width = max([r.shape[1] for r in subregions if r.size > 0], default=0)
    total_height = sum([r.shape[0] for r in subregions if r.size > 0]) + gap * (len(subregions) - 1)
    
    if max_width == 0 or total_height == 0:
        return None
    
    blank_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    
    y_offset = 0
    for region in subregions:
        if region.size == 0:
            continue
        h, w = region.shape[:2]
        blank_image[y_offset:y_offset+h, :w] = region
        y_offset += h + gap
    
    return blank_image


def point_line_distance(x0, y0, x1, y1, x2, y2):
    """Calculate the shortest distance from a point (x0, y0) to a line segment (x1, y1) - (x2, y2)."""
    A = x0 - x1
    B = y0 - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    return ((x0 - xx) ** 2 + (y0 - yy) ** 2) ** 0.5, (xx, yy)

def find_closest_lines(img, point):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        print("No lines detected.")
        return

    x0, y0 = point
    closest_vertical = None
    closest_horizontal = None
    min_v_dist = 10000
    min_h_dist = 10000

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate the minimum distance from the point to the line
        dist, _ = point_line_distance(x0, y0, x1, y1, x2, y2)
        
        # Check if the line is mostly vertical
        if abs(x1 - x2) < abs(y1 - y2):  
            if dist < min_v_dist:
                min_v_dist = dist
                closest_vertical = (x1, y1, x2, y2)
        
        # Check if the line is mostly horizontal
        else:  
            if dist < min_h_dist:
                min_h_dist = dist
                closest_horizontal = (x1, y1, x2, y2)
    
    return closest_vertical, closest_horizontal, min_v_dist, min_h_dist

def point_to_line_distance(px, py, x0, y0, x1, y1):
    """Calculate the perpendicular distance from point (px, py) to line (x0, y0) - (x1, y1)."""
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    
    # Perpendicular distance formula
    distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
    return distance

def get_character_boxes(image, point_0):
    X0, Y0 = point_0
    image = line_remove(image)
    
    H, W = image.shape[0:2]
    blank_image = np.zeros((H, W), dtype=np.uint8) + 255
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xc, yc, points = [], [], []
    rect = []
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 40>w and 40>h and (abs(x+w//2-X0)>5 or abs(y+h//2-Y0)>5):
            # xc.append(int(x+w/2))
            # yc.append(int(y+h/2))
            # points.append((int(x+w/2), int(y+h/2)))
            blank_image[y:y+h, x:x+w] = 0

            # cv2.rectangle(imgCpy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
    
    # get contour in blank image
    blank_image = cv2.erode(blank_image, np.ones((13, 13)), iterations=1)  
    _, thresh = cv2.threshold(blank_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgCpy = image.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_blank_image = np.zeros((H, W), dtype=np.uint8) + 255
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>25 and h>25:
            xc.append(int(x+w/2))
            yc.append(int(y+h/2))
            points.append((int(x+w/2), int(y+h/2)))
            rect.append([x,y,int(x+w), int(y+h)])

            cv2.rectangle(imgCpy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Green rectangle    
    return xc, yc, points, rect

def point_line_rect(x0, y0, rects):
    dists = []
    for rect in rects:
        left_x, top_y, right_x, bottom_y = rect
        lines = [
            [left_x, top_y, right_x, top_y],
            [left_x, top_y, left_x, bottom_y],
            [left_x, bottom_y, right_x, bottom_y],
            [right_x, top_y, right_x, bottom_y]]
        minDist = 10000
        for x1, y1, x2, y2 in lines:
            """Calculate the shortest distance from a point (x0, y0) to a line segment (x1, y1) - (x2, y2)."""
            A = x0 - x1
            B = y0 - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D
            param = dot / len_sq if len_sq != 0 else -1

            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dist = ((x0 - xx) ** 2 + (y0 - yy) ** 2) ** 0.5
            if dist < minDist: minDist = dist
        dists.append(minDist)
    return dists
def find_closest_and_reference_points(specific_point, points, rects):
    X0, Y0 = specific_point
    distances = point_line_rect(X0, Y0, rects)
    
    # Compute distances from the specific point to all other points
    distances_ = np.linalg.norm(points - np.array([X0, Y0]), axis=1)
    
    # Find the closest point
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    closest_distance = distances[closest_idx]
    
    # Define reference distance
    reference_distance = closest_distance +15
    
    # Get reference points within reference distance
    reference_points = points[distances <= reference_distance]
    
    return closest_point, reference_points

def find_interest_points(reference_points, all_points, maxV=90, minV=35):
    interest_points = []
    for ref_point in reference_points:
        ref_x, ref_y = ref_point
        
        # Condition for interest points
        condition = (
            ((np.abs(all_points[:, 0] - ref_x) < 35) & (np.abs(all_points[:, 1] - ref_y) < maxV)) |
            ((np.abs(all_points[:, 0] - ref_x) < 90) & (np.abs(all_points[:, 1] - ref_y) < minV))
        )
        
        interest_points.extend(all_points[condition].tolist())
    
    return np.unique(interest_points, axis=0)  # Remove duplicates

def get_interest_points_indices(all_points, interest_points):
    indices = [np.where((all_points == ip).all(axis=1))[0][0] for ip in interest_points]
    return indices

def getRoiInfo(interest_points_indices, rectInfo, circled_number_point, size=20):
    interest_rectInfo = rectInfo[interest_points_indices]
    minX, minY, maxX, maxY = min(interest_rectInfo[:, 0]), min(interest_rectInfo[:, 1]), max(interest_rectInfo[:, 2]), max(interest_rectInfo[:, 3])
    minX, minY, maxX, maxY = min(minX, max(circled_number_point[0]-size, 0)), min(minY, max(circled_number_point[1]-size, 0)), max(maxX, circled_number_point[0]+size), max(maxY, circled_number_point[1]+size)
    
    return [int(minX), max(0, int(minY-8)), int(maxX), int(maxY+8)]

def integrate_rects(ROIs, img_height=600, gap=5):
    # Calculate the height of each rectangle
    sizes = [(roi.shape[1], roi.shape[0]) for roi in ROIs]
    positions = rpack.pack(sizes, max_height=700)
    # Determine total height
    total_height = max(y + h for (x, y), (w, h) in zip(positions, sizes))
    total_width = max(x+w for (x, y), (w, h) in zip(positions, sizes))
    # Create a blank white image
    integrated_img = np.zeros((total_height, total_width, 3), dtype=np.uint8) 
    # integrated_img[:, :] = [0, 0, 255]
    # Draw the rectangles with black borders
    cnt = 0
    for (x, y), (w, h) in zip(positions, sizes):
        integrated_img[y:y+h, x:x+w] = ROIs[cnt]
        cnt += 1

    return integrated_img

def newGetRoi(img, balloon_numbers, roi_path):
    ROIs = []
    removeS = 30
    padding = 300    
    roi_padding = 25
    for balloon_number in balloon_numbers:
        number, point = list(balloon_number.items())[0]
        if number == 0: continue
        # remove other circled number
        imgCpy = img.copy()
        for b in balloon_numbers:
            if number != list(b.items())[0][0]:
                tempPoint = list(b.items())[0][1]
                imgCpy[tempPoint[1]-removeS:tempPoint[1]+removeS,tempPoint[0]-removeS:tempPoint[0]+removeS] = [255, 255, 255]
        
        croppedImg = imgCpy[max(0,point[1]-padding):(point[1]+padding), max(0,point[0]-padding):(point[0]+padding)]        
        point_in_cropped = (padding, padding)
        if point[0]-padding<0:
            point_in_cropped = (point[0], padding)
        if point[1]-padding<0:
            point_in_cropped = (padding, point[1])  

        _, _, points, rectInfo = get_character_boxes(croppedImg, point_in_cropped)
        all_points = np.array(points)
        rectInfo = np.array(rectInfo)
        closest_point, reference_points = find_closest_and_reference_points(point_in_cropped, all_points, rectInfo)
        # interest_points = find_interest_points(reference_points, all_points, maxV=90)
        interest_points_indices = get_interest_points_indices(all_points, reference_points)

        roi_rect = getRoiInfo(interest_points_indices, rectInfo, point_in_cropped)
        roi_img = croppedImg[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2]]
        padded_roi = cv2.copyMakeBorder(roi_img, roi_padding, roi_padding, roi_padding, roi_padding, cv2.BORDER_CONSTANT, value=(0,0,0))
        ROIs.append(padded_roi)

    integraded_roi = integrate_rects(ROIs)  
    cv2.imwrite(roi_path, integraded_roi)
    return
def getRoi(img, balloon_numbers, roi_path, Isdraw=False):
    ROIs = []
    removeS = 30
    ref_Diff = 50
    padding = 300    
    for balloon_number in balloon_numbers:
        number, point = list(balloon_number.items())[0]
        # remove other circled number
        imgCpy = img.copy()
        for b in balloon_numbers:
            if number != list(b.items())[0][0]:
                tempPoint = list(b.items())[0][1]
                imgCpy[tempPoint[1]-removeS:tempPoint[1]+removeS,tempPoint[0]-removeS:tempPoint[0]+removeS] = [255, 255, 255]
        
        croppedImg = imgCpy[max(0,point[1]-padding):(point[1]+padding), max(0,point[0]-padding):(point[0]+padding)]
        # get the closest horizontal line and vertical line
        point_in_cropped = (padding, padding)
        if point[0]-padding<0:
            point_in_cropped = (point[0], padding)
        if point[1]-padding<0:
            point_in_cropped = (padding, point[1])            

        vertical_line, horizontal_line, minV, minH = find_closest_lines(croppedImg, point_in_cropped)
    
        # get the closest text
        # text, left, top, w, h, accu, xc, yc = getting_textdata(croppedImg, '--psm 6')
        xc, yc = get_character_boxes(croppedImg.copy())
        distances_to_point = [np.sqrt((xc[i] - point_in_cropped[0])**2 + (yc[i] - point_in_cropped[1])**2) for i in range(len(xc))]
        sorted_indices = np.argsort(distances_to_point)
        min_index = sorted_indices[1]
        closest_text_center = (xc[min_index], yc[min_index])        
        
        if Isdraw:
            croppedImgCpy = croppedImg.copy()
            if vertical_line:
                cv2.line(croppedImgCpy, (vertical_line[0], vertical_line[1]), (vertical_line[2], vertical_line[3]), (0, 255, 0), 2)

            if horizontal_line:
                cv2.line(croppedImgCpy, (horizontal_line[0], horizontal_line[1]), (horizontal_line[2], horizontal_line[3]), (255, 0, 0), 2)
            cv2.imwrite("draw_line.jpg", croppedImgCpy)

        # comparision distances from point about every horizontal line and vertical line. 
        if abs(minV-minH) > ref_Diff:
            if minV < minH: 
                X, H = vertical_line[0], abs(vertical_line[1]-vertical_line[3])


                if abs(point_in_cropped[0]-min(X, closest_text_center[0])) < 50:
                    if point_in_cropped[0]>X:
                        Roi = croppedImg[:, min(X, closest_text_center[0])-85: point_in_cropped[0]+20]
                    else:
                        Roi = croppedImg[:, point_in_cropped[0]-20:max(X, closest_text_center[0])|+85]
                else:
                    # Roi = croppedImg[:, max(0, min(point_in_cropped[0],X)-20):max(point_in_cropped[0],X)]

                    if point_in_cropped[0] < X:
                        if X> closest_text_center[0]:
                            Roi = croppedImg[:,point_in_cropped[0]-20:X]
                        else:
                            Roi = croppedImg[:,point_in_cropped[0]-20:closest_text_center[0] + 85]
                    else:
                        if X> closest_text_center[0]: 
                            Roi = croppedImg[:,max(0, closest_text_center[0]-85):point_in_cropped[0]+20]
                        else: 
                            Roi = croppedImg[:,X:point_in_cropped[0]+20]
                            
                if point_in_cropped[0] < X:
                    Roi = cv2.rotate(Roi, cv2.ROTATE_90_CLOCKWISE)
                else:
                    Roi = cv2.rotate(Roi, cv2.ROTATE_90_COUNTERCLOCKWISE)                    
            else: 
                Y, W = horizontal_line[1], abs(horizontal_line[0]-horizontal_line[2])
                # Roi = croppedImg[min(point_in_cropped[1],Y):max(point_in_cropped[1],Y)]



                if abs(point_in_cropped[1]-min(Y, closest_text_center[1])) < 50:
                    if point_in_cropped[1]>Y:
                        Roi = croppedImg[min(Y, closest_text_center[1])-85: point_in_cropped[1]+20]
                    else:
                        Roi = croppedImg[point_in_cropped[1]-20:max(Y, closest_text_center[1])|+85]
                else:
                    # Roi = croppedImg[:, max(0, min(point_in_cropped[0],X)-20):max(point_in_cropped[0],X)]

                    if point_in_cropped[1] < Y:
                        if Y> closest_text_center[1]:
                            Roi = croppedImg[point_in_cropped[1]-20:Y]
                        else:
                            Roi = croppedImg[point_in_cropped[1]-20:closest_text_center[1] + 85]
                    else:
                        if Y> closest_text_center[1]:
                            Roi = croppedImg[max(0, closest_text_center[1]-85):point_in_cropped[1]+20]
                        else:
                            Roi = croppedImg[Y:point_in_cropped[1]+20]

        else:
            # Compute distances to lines
            dist_lineV = point_to_line_distance(closest_text_center[0], closest_text_center[1], *vertical_line)
            dist_lineH = point_to_line_distance(closest_text_center[0], closest_text_center[1], *horizontal_line)

            
            if dist_lineV + minV < dist_lineH + minH:
                X, H = vertical_line[0], abs(vertical_line[1]-vertical_line[3])

                if abs(point_in_cropped[0]-min(X, closest_text_center[0])) < 50:
                    if point_in_cropped[0]>X:
                        Roi = croppedImg[:, min(X, closest_text_center[0])-85: point_in_cropped[0]+20]
                    else:
                        Roi = croppedImg[:, point_in_cropped[0]-20:max(X, closest_text_center[0])|+85]
                else:
                    # Roi = croppedImg[:, max(0, min(point_in_cropped[0],X)-20):max(point_in_cropped[0],X)]

                    if point_in_cropped[0] < X:
                        if X> closest_text_center[0]:
                            Roi = croppedImg[:,point_in_cropped[0]-20:X]
                        else:
                            Roi = croppedImg[:,point_in_cropped[0]-20:closest_text_center[0] + 85]
                    else:
                        if X> closest_text_center[0]: Roi = croppedImg[:,max(0, closest_text_center[0]-85):point_in_cropped[0]+20]
                        else: Roi = croppedImg[:,X:point_in_cropped[0]+20]

                if point_in_cropped[0] < X:
                    Roi = cv2.rotate(Roi, cv2.ROTATE_90_CLOCKWISE)
                else:
                    Roi = cv2.rotate(Roi, cv2.ROTATE_90_COUNTERCLOCKWISE)      
            else:
                Y, W = horizontal_line[1], abs(horizontal_line[0]-horizontal_line[2])
                # Roi = croppedImg[min(point_in_cropped[1],Y):max(point_in_cropped[1],Y)]


                if abs(point_in_cropped[1]-min(Y, closest_text_center[1])) < 50:
                    if point_in_cropped[1]>Y:
                        Roi = croppedImg[min(Y, closest_text_center[1])-85: point_in_cropped[1]+20]
                    else:
                        Roi = croppedImg[point_in_cropped[1]-20:max(Y, closest_text_center[1])|+85]
                else:
                    # Roi = croppedImg[:, max(0, min(point_in_cropped[0],X)-20):max(point_in_cropped[0],X)]
                    if point_in_cropped[1] < Y:
                        if Y> closest_text_center[1]:
                            Roi = croppedImg[point_in_cropped[1]-20:Y]
                        else:
                            Roi = croppedImg[point_in_cropped[1]-20:closest_text_center[1] + 85]
                    else:
                        if Y> closest_text_center[1]:
                            Roi = croppedImg[max(0, closest_text_center[1]-85):point_in_cropped[1]+20]
                        else:
                            Roi = croppedImg[Y:point_in_cropped[1]+20]
                            
        
        ROIs.append(Roi)
    gap = 10
    total_height = sum(roi.shape[0] for roi in ROIs) + (len(ROIs) - 1) * gap
    width = padding * 2
    blank_image = np.zeros((total_height, width, 3), dtype=np.uint8)
    y_offset = 0
    for roi in ROIs:
        h = roi.shape[0]  # Get height of current ROI
        blank_image[y_offset:y_offset + h, :width] = roi  # Place ROI
        y_offset += h + gap  # Move down for next ROI

    cv2.imwrite(roi_path, blank_image)

    return 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_available_key():
    """Rotates API keys to avoid hitting rate limits"""
    return next(api_key_cycle)

def extractDimension(image_path):
    api_key = get_available_key()
    # print("API_KEY: ", api_key)
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": 'Give base size and tolerances for all circled number as json. ex: {"1":{"dim":"28","tolerance":"0.05/0"}, "2":{"dim":"⌀10","tolerance":"0/-0.05"}...}.'
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 1500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())

    if response.status_code == 200:
        res_json = response.json()
        json_string = res_json["choices"][0]["message"]["content"][8:-4].replace("\n", "")
    
        return json_string
    else:
        return ""