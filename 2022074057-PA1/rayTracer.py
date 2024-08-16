#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 
class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma;
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

class Sphere:
    def __init__(self, center, radius, shader):
        self.center = center
        self.radius = radius
        self.shader = shader
    
    def intersect(self, rayOrig, rayDirect):
        originToCenter = rayOrig - self.center
        a = np.dot(rayDirect, rayDirect)
        b = 2.0 * np.dot(originToCenter, rayDirect)
        c = np.dot(originToCenter, originToCenter) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            if t1 > 0 and t2 > 0: # if there exists intersection
                intersectDist = min(t1, t2)
                intersectPoint = rayOrig + intersectDist * rayDirect
                normal = (intersectPoint - self.center) / self.radius
                return intersectDist, normal
        return -1, np.array([0, 0, 0]) # if no intersection

class Camera:
    def __init__(self, viewPoint, viewDir, viewProjNormal, viewUp, viewWidth, viewHeight, projDistance):
        self.viewPoint = viewPoint
        self.viewDir = viewDir
        self.viewProjNormal = viewProjNormal
        self.viewUp = viewUp
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight
        self.projDistance = projDistance

class Shader:
    def __init__(self, name, type, diffuseColor):
        self.name = name
        self.type = type
        self.diffuseColor = diffuseColor

class Lambertian(Shader): # child class of Shader class
    def __init__(self, name, type, diffuseColor):
        super().__init__(name, type, diffuseColor)

class Phong(Shader): # child class of Shader class with additional parameters
    def __init__(self, name, type, diffuseColor, specularColor, exponent):
        super().__init__(name, type, diffuseColor)
        self.specularColor = specularColor
        self.exponent = exponent

def normalize(vector): # normalize a vector to unit length
    return vector / np.sqrt(vector @ vector)

def rayTracing(rayOrigin, rayDirect, surfaceList):
    closestDist = sys.maxsize # distance to intersection point
    closestIndex = -1 # no intersection found yet

    count = 0
    for surface in surfaceList:
        dist, _ = surface.intersect(rayOrigin, rayDirect)
        if dist > 0 and dist < closestDist: # if distance is closer than the current closestDist, update values
            closestDist = dist
            closestIndex = count
        count += 1

    return closestIndex, closestDist

def shading(dist, closestIndex, rayDirect, viewPoint, surfaceList, lightList):
    if closestIndex == -1:
        return np.array([0, 0, 0])
    
    surface = surfaceList[closestIndex]
    intersectPoint = viewPoint + dist * rayDirect
    n = surface.intersect(viewPoint, rayDirect)[1]  # normal at intersection point

    r = 0 # red
    g = 0 # green
    b = 0 # blue

    for lightPosition, lightIntensity in lightList:
        lightDirect = normalize(lightPosition - intersectPoint)
        _, shadowDist = rayTracing(intersectPoint, lightDirect, surfaceList)
        if shadowDist > 0 and shadowDist < np.linalg.norm(lightPosition - intersectPoint): # intersection point is in shadow, so continue
            continue
        
        if surface.shader.__class__.__name__ == 'Lambertian':
            diffuse = np.dot(lightDirect, n) # angle between light & normal (how light diffused)
            if diffuse > 0:
                r += surface.shader.diffuseColor[0] * lightIntensity[0] * diffuse
                g += surface.shader.diffuseColor[1] * lightIntensity[1] * diffuse
                b += surface.shader.diffuseColor[2] * lightIntensity[2] * diffuse

        elif surface.shader.__class__.__name__ == 'Phong':
            v = normalize(viewPoint - intersectPoint) # vector from intersection point to viewer's position
            h = normalize(lightDirect + v) # half-vector btw light direction and view direction
            diffuse = np.dot(n, lightDirect)
            specular = np.dot(n, h) ** surface.shader.exponent # shininess of surface
            if diffuse > 0:
                r += surface.shader.diffuseColor[0] * lightIntensity[0] * diffuse
                g += surface.shader.diffuseColor[1] * lightIntensity[1] * diffuse
                b += surface.shader.diffuseColor[2] * lightIntensity[2] * diffuse
            if specular > 0:
                r += surface.shader.specularColor[0] * lightIntensity[0] * specular
                g += surface.shader.specularColor[1] * lightIntensity[1] * specular
                b += surface.shader.specularColor[2] * lightIntensity[2] * specular

    result = Color(r, g, b)
    result.gammaCorrect(2.2)
    return result.toUINT8()

def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # set default values
    viewDir = np.array([0,0,-1]).astype(np.float64)
    viewUp = np.array([0,1,0]).astype(np.float64)
    viewProjNormal = -1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth = 1.0
    viewHeight = 1.0
    projDistance = 1.0
    intensity = np.array([1,1,1]).astype(np.float64)  # how bright the light is
    # print(np.cross(viewDir, viewUp))

    imgSize = np.array(root.findtext('image').split()).astype(np.int32)

    # get information of camera from XML file
    for c in root.findall('camera'):
        viewPoint = np.array(c.findtext('viewPoint').split()).astype(np.float64)
        viewDir = np.array(c.findtext('viewDir').split()).astype(np.float64)
        viewUp = np.array(c.findtext('viewUp').split()).astype(np.float64)
        viewProjNormal = np.array(c.findtext('projNormal').split()).astype(np.float64)
        viewWidth = np.array(c.findtext('viewWidth').split()).astype(np.float64)
        viewHeight = np.array(c.findtext('viewHeight').split()).astype(np.float64)
        if(c.findtext('projDistance')):
            projDistance = np.array(c.findtext('projDistance').split()).astype(np.float64)
    camera = Camera(viewPoint, viewDir, viewProjNormal, viewUp, viewWidth, viewHeight, projDistance)

    # get information of shader from XML file
    shaderList = []
    for c in root.findall('shader'):
        diffuse = np.array(c.findtext('diffuseColor').split()).astype(np.float64)
        shaderName = c.get('name')
        shaderType = c.get('type')

        if(shaderType == 'Lambertian'):
            shader = Lambertian(shaderName, shaderType, diffuse)
            shaderList.append(shader)
        elif(shaderType == 'Phong'):
            specular = np.array(c.findtext('specularColor').split()).astype(np.float64)
            exponent = np.array(c.findtext('exponent').split()).astype(np.float64)[0]
            shader = Phong(shaderName, shaderType, diffuse, specular, exponent)
            shaderList.append(shader)
    #code.interact(local=dict(globals(), **locals()))  

    # create figures
    surfaceList = []
    for c in root.findall('surface'):
        surfaceType = c.get('type')
        ref = ''
        if surfaceType == 'Sphere':
            center = np.array(c.findtext('center').split()).astype(np.float64)
            radius = np.array(c.findtext('radius')).astype(np.float64)
            for i in c:
                if(i.tag == 'shader'):
                    ref = i.get('ref')
                    for j in shaderList:
                        if (j.name == ref):
                            surfaceList.append(Sphere(center, radius, j))
                            break

    # get information of light from XML file
    lightList = []
    for c in root.findall('light'):
        position = np.array(c.findtext('position').split()).astype(np.float64)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float64)
        lightList.append((position, intensity))

    # Create an empty image
    channels=3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:]=0

    # Create image as defined
    pixelX = camera.viewWidth / imgSize[0]
    pixelY = camera.viewHeight / imgSize[1]

    wVec = camera.viewProjNormal
    uVec = np.cross(wVec, camera.viewUp)
    vVec = np.cross(wVec, uVec)
    
    wUnit = normalize(wVec)
    uUnit = normalize(uVec)
    vUnit = normalize(vVec)
    
    imgVector = wVec - camera.projDistance * wUnit # vector between image plane and origin

    # calculate starting point
    uStart = (camera.viewWidth / (2 * imgSize[0])) * (imgSize[0] + 1)
    vStart = (camera.viewHeight / (2 * imgSize[1])) * (imgSize[1] + 1)
    start = (camera.viewDir + imgVector) + (uStart * uUnit) - (vStart * vUnit)

    for x in np.arange(imgSize[0]): # for each pixel in image,
        for y in np.arange(imgSize[1]):
            # use rayTracing and shading to shade pixel based on closest surface & distance
            rayDirect = start - (pixelX * x * uUnit) + (pixelY * y * vUnit)
            closestIndex, closestDist = rayTracing(camera.viewPoint, rayDirect, surfaceList)
            img[y][x] = shading(closestDist, closestIndex, rayDirect, camera.viewPoint, surfaceList, lightList)
    
    rawimg = Image.fromarray(img, 'RGB')
    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__=="__main__":
    main()