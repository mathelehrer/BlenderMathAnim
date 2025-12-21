"""
 here are constants that are subject to change in blender versions
    you should only code with these constants to allow for easier updates
"""

import bpy

def blender_version():
    return bpy.app.version

print("blender version: ",blender_version())
# blender 3
if blender_version()<(4,0):
    TRANSMISSION = 'Transmission'
    SPECULAR = 'Specular'
    EMISSION ='Emission'
else:
    TRANSMISSION = 'Transmission Weight'
    SPECULAR = 'Specular IOR Level'
    EMISSION = 'Emission Color'

CYCLES="CYCLES"
if blender_version()<(4,2):
    BLENDER_EEVEE='BLENDER_EEVEE'
elif blender_version()<(5,0):
    BLENDER_EEVEE='BLENDER_EEVEE_NEXT'
else:
    BLENDER_EEVEE='BLENDER_EEVEE'