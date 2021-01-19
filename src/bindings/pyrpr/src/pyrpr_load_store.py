#**********************************************************************
# Copyright 2020 Advanced Micro Devices, Inc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#********************************************************************
import bpy

import dataclasses
import numpy as np
import platform

from _pyrpr_load_store import ffi
import pyrpr

lib = None


def init(rpr_sdk_bin_path):

    global lib

    path = get_library_path(rpr_sdk_bin_path)
    lib = ffi.dlopen(path)


def get_library_path(rpr_sdk_bin_path):

    os = platform.system()

    if os == "Windows":
        return str(rpr_sdk_bin_path / 'RprLoadStore64.dll')
    elif os == "Linux":
        return str(rpr_sdk_bin_path / 'libRprLoadStore64.so')
    elif os == "Darwin":
        return str(rpr_sdk_bin_path / 'libRprLoadStore64.dylib')
    else:
        assert False


"""
extern RPR_API_ENTRY rpr_status rprsAssignShapeToGroup(rpr_shape shape, const rpr_char * groupName);
extern RPR_API_ENTRY rpr_status rprsAssignLightToGroup(rpr_light light, const rpr_char * groupName);
extern RPR_API_ENTRY rpr_status rprsAssignCameraToGroup(rpr_camera camera, const rpr_char * groupName);
extern RPR_API_ENTRY rpr_status rprsAssignParentGroupToGroup(const rpr_char * groupChild, const rpr_char * groupParent);
extern RPR_API_ENTRY rpr_status rprsSetTransformGroup(const rpr_char * groupChild, const float * matrixComponents);

// structSize : size of this struct in Byte (internally used to identify if different versions)
// interpolationType : unused for now - set it to 0
//
// example : if the animation has 2 FLOAT3 defining translation at time 0.5 and 3.0  for a translation along y axis , we have  :
//
//nbTimeKeys = 2
//nbTransformValues = 2 
//timeKeys        = { 0.5 , 3.0 }
//transformValues = { 0.0,0.0,0.0,  0.0,1.0,0.0,  }
struct __rprs_animation
{
    rpr_uint structSize;
    char * groupName;
    rprs_animation_movement_type movementType;
    rpr_uint interpolationType;
    rpr_uint nbTimeKeys;
    rpr_uint nbTransformValues;
    float * timeKeys;
    float * transformValues;
};

#extern RPR_API_ENTRY rpr_status rprsAddAnimation(const rprs_animation * anim);
"""


MOVEMENT_CONSTANTS = {  # animation track ID
    'translation': 1,  # RPRS_ANIMATION_MOVEMENTTYPE_TRANSLATION
    'rotation': 2,  # RPRS_ANIMATION_MOVEMENTTYPE_ROTATION
    'scale': 3,  # RPRS_ANIMATION_MOVEMENTTYPE_SCALE
}


DATA_SIZE = {  # number of floats used to store value
    'translation': 3,
    'rotation': 4,
    'scale': 3,
}


class Object:
    core_type_name = 'void*'

    def __init__(self, core_type_name=None):
        self.ffi_type_name = (core_type_name if core_type_name is not None else self.core_type_name) + '*'
        self._reset_handle()

    def __init__(self, core_type_name, obj):
        self.ffi_type_name = (core_type_name if core_type_name is not None else self.core_type_name) + '*'
        self._handle_ptr = ffi.cast(self.ffi_type_name, obj._handle_ptr)

    def _reset_handle(self):
        self._handle_ptr = ffi.new(self.ffi_type_name, ffi.NULL)

    def _get_handle(self):
        return self._handle_ptr[0]


class ArrayObject:
    def __init__(self, core_type_name, init_data):
        self._handle_ptr = ffi.new(core_type_name, init_data)

    def __del__(self):
        del self._handle_ptr


@dataclasses.dataclass
class Animation:
    struct_size: int
    group_name: str
    interpolation_type: int
    time_keys_number: int
    transform_keys_number: int
    movement_type: int = 1
    time_keys: ArrayObject = None
    transform_values: ArrayObject = None

    def __init__(self, group_name, category, keys, data):
        self.group_name = group_name
        self.movement_type = category
        self.time_keys = keys[:]
        self.time_keys_number = len(keys)
        self.transform_values = data
        self.transform_keys_number = len(data)

        # pack data to CFFI representation
        assert self.time_keys_number == self.transform_keys_number, \
            f"Number of keyframe keys {self.time_keys_number} is not equal to number of values {self.transform_keys_number}!"

        self.group_name_data = ffi.new('char[]', pyrpr.encode(self.group_name))

        self.values_bytes_number = self.transform_keys_number * DATA_SIZE[self.movement_type]

        self.interpolation_type = 0
        self.movement_type = MOVEMENT_CONSTANTS[category]

        keytimes_in_seconds = [frame/bpy.context.scene.render.fps for frame in self.time_keys]

        self.keys = np.ascontiguousarray(keytimes_in_seconds[:], dtype=np.float32)
        self.keys_array = ffi.cast("float*", self.keys.ctypes.data)

        # turn list of tuples to flat list
        self.values = np.ascontiguousarray(self.transform_values[:], dtype=np.float32).reshape(self.values_bytes_number)
        self.values_array = ffi.cast("float*", self.values.ctypes.data)

        self.packed_data = ffi.new("rprs_animation*")
        self.struct_size = ffi.sizeof("rprs_animation")

        self.packed_data.structSize = self.struct_size
        self.packed_data.groupName = self.group_name_data
        self.packed_data.movementType = self.movement_type
        self.packed_data.interpolationType = self.interpolation_type
        self.packed_data.nbTimeKeys = self.time_keys_number
        self.packed_data.nbTransformValues = self.time_keys_number
        self.packed_data.timeKeys = self.keys_array
        self.packed_data.transformValues = self.values_array


def rprs_add_extra_camera(rpr_camera):
    return lib.rprsAddExtraCamera(rpr_camera._get_handle())


def rprs_assign_shape_to_group(shape: pyrpr.Shape, group_name: str):
    return lib.rprsAssignShapeToGroup(shape._get_handle(), pyrpr.encode(group_name))


def rprs_assign_light_to_group(light: pyrpr.Light, group_name: str):
    return lib.rprsAssignLightToGroup(light._get_handle(), pyrpr.encode(group_name))


def rprs_assign_camera_to_group(camera: pyrpr.Camera, group_name: str):
    return lib.rprsAssignCameraToGroup(camera._get_handle(), pyrpr.encode(group_name))


def rprs_assign_parent_group_to_group(group_name: str, parent_name: str):
    return lib.rprsAssignParentGroupToGroup(pyrpr.encode(group_name), pyrpr.encode(parent_name))


def rprs_set_transform_to_group(group_name: str, transform):
    return lib.rprsSetTransformGroup(pyrpr.encode(group_name), transform)


def rprs_add_animation(animation: Animation):
    return lib.rprsAddAnimation(animation)


def export(name, context, scene, flags):
    # last param defines export bit flags.
    # image handling type flags:
    # RPRLOADSTORE_EXPORTFLAG_EXTERNALFILES (1 << 0) - image data will be stored to rprsb external file
    # RPRLOADSTORE_EXPORTFLAG_COMPRESS_IMAGE_LEVEL_1 (1 << 1) - image data will be lossless compressed during export
    # RPRLOADSTORE_EXPORTFLAG_COMPRESS_IMAGE_LEVEL_2 (1 << 2) - image data will be lossy compressed during export
    #  note: without any of above flags images will not be exported.
    return lib.rprsExport(pyrpr.encode(name), context._get_handle(), scene._get_handle(),
                          0, ffi.NULL, ffi.NULL, 0, ffi.NULL, ffi.NULL, flags)
