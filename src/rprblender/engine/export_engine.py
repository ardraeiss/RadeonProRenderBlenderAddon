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
"""
Scene export to file
"""
import math
import numpy as np

import pyrpr
import pyrpr_load_store

from rprblender.export import (
    instance,
    object,
    particle,
    world,
    camera
)
from .context import RPRContext, RPRContext2
from .engine import Engine
import pyrpr

from rprblender.utils.logging import Log
log = Log(tag='ExportEngine')


class ExportEngine(Engine):
    TYPE = 'EXPORT'

    def __init__(self):
        self.rpr_context = RPRContext()
        self.rpr_context.engine_type = self.TYPE

    def sync(self, context):
        """ Prepare scene for export """
        log('Start sync')

        depsgraph = context.evaluated_depsgraph_get()
        self.rpr_context.blender_data['depsgraph'] = depsgraph
        scene = depsgraph.scene

        use_contour = scene.rpr.is_contour_used()

        scene.rpr.init_rpr_context(self.rpr_context, use_contour_integrator=use_contour)

        self.rpr_context.scene.set_name(scene.name)
        self.rpr_context.width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
        self.rpr_context.height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

        world.sync(self.rpr_context, scene.world)

        # camera, objects, particles
        for obj in self.depsgraph_objects(depsgraph, with_camera=True):
            indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)
            object.sync(self.rpr_context, obj, indirect_only=indirect_only,
                        frame_current=scene.frame_current)

        # instances
        for inst in self.depsgraph_instances(depsgraph):
            indirect_only = inst.parent.original.indirect_only_get(view_layer=depsgraph.view_layer)
            instance.sync(self.rpr_context, inst, indirect_only=indirect_only,
                          frame_current=scene.frame_current)

        # rpr_context parameters
        self.rpr_context.set_parameter(pyrpr.CONTEXT_PREVIEW, False)
        scene.rpr.export_ray_depth(self.rpr_context)

        # EXPORT CAMERA
        camera_key = object.key(scene.camera)  # current camera key
        rpr_camera = self.rpr_context.create_camera(camera_key)
        self.rpr_context.scene.set_camera(rpr_camera)
        camera_obj = depsgraph.objects.get(camera_key, None)
        if not camera_obj:
            camera_obj = scene.camera

        camera_data = camera.CameraData.init_from_camera(camera_obj.data, camera_obj.matrix_world,
                                                         self.rpr_context.width / self.rpr_context.height)
        camera_data.export(rpr_camera)

        # sync Motion Blur
        self.rpr_context.do_motion_blur = scene.render.use_motion_blur and \
                                          not math.isclose(scene.camera.data.rpr.motion_blur_exposure, 0.0)

        if self.rpr_context.do_motion_blur:
            self.sync_motion_blur(depsgraph)
            rpr_camera.set_exposure(scene.camera.data.rpr.motion_blur_exposure)
            self.set_motion_blur_mode(scene)

        # adaptive subdivision will be limited to the current scene render size
        self.rpr_context.enable_aov(pyrpr.AOV_COLOR)
        self.rpr_context.sync_auto_adapt_subdivision()

        self.rpr_context.sync_portal_lights()

        # Exported scene will be rendered vertically flipped, flip it back
        self.rpr_context.set_parameter(pyrpr.CONTEXT_Y_FLIP, True)

        log('Finish sync')

    def _set_scene_frame(self, scene, frame, subframe=0.0):
        scene.frame_set(frame, subframe=subframe)

    def export_to_rpr(self, filepath: str, flags):
        """
        Export scene to RPR file
        :param filepath: full output file path, including filename extension
        """
        log('export_to_rpr')
        pyrpr_load_store.export(filepath, self.rpr_context.context, self.rpr_context.scene, flags)


class ExportEngine2(ExportEngine):
    TYPE = 'EXPORT'

    def __init__(self):
        self.rpr_context = RPRContext2()
        self.rpr_context.engine_type = self.TYPE


class ExportEngineAnimated(ExportEngine2):
    type = 'EXPORT'

    def __init__(self):
        super().__init__()
        self.global_matrix = np.identity

    def sync(self, context):
        log('Start sync')

        depsgraph = context.evaluated_depsgraph_get()
        self.rpr_context.blender_data['depsgraph'] = depsgraph
        scene = depsgraph.scene

        use_contour = scene.rpr.is_contour_used()

        scene.rpr.init_rpr_context(self.rpr_context, use_contour_integrator=use_contour)

        self.rpr_context.scene.set_name(scene.name)
        self.rpr_context.width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
        self.rpr_context.height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

        world.sync(self.rpr_context, scene.world)

        # collect animation data
        orig_frame = scene.frame_current
        try:
            keyframes = self.collect_keyframes(scene)
            animation_data = self.collect_animation_data_by_keyframes(context, keyframes)
            log.info(f"keyframes:\n{keyframes}")
            log.info(f"animation_data:\n{animation_data}")
        finally:
            scene.frame_set(orig_frame)

        self.sync_animated_objects(depsgraph, with_camera=False)

        # EXPORT CAMERA
        camera_key = object.key(scene.camera)  # current camera key
        rpr_camera = self.rpr_context.create_camera(camera_key)
        self.rpr_context.scene.set_camera(rpr_camera)
        camera_obj = depsgraph.objects.get(camera_key, None)
        if not camera_obj:
            camera_obj = scene.camera

        camera_group_name, camera_parent_group_name = self.get_group_names(camera_obj)
        self.assing_group_to_object(rpr_camera, camera_group_name)
        if camera_obj.parent:
            pyrpr_load_store.rprs_assign_parent_group_to_group(camera_group_name, camera_parent_group_name)

        pyrpr_load_store.rprs_set_transform_to_group(camera_group_name, self.get_local_matrix(camera_obj))

        camera_data = camera.CameraData.init_from_camera(camera_obj.data, camera_obj.matrix_world,
                                                         self.rpr_context.width / self.rpr_context.height)
        camera_data.export(rpr_camera)

        # sync Motion Blur
        self.rpr_context.do_motion_blur = scene.render.use_motion_blur and \
                                          not math.isclose(scene.camera.data.rpr.motion_blur_exposure, 0.0)

        if self.rpr_context.do_motion_blur:
            self.sync_motion_blur(depsgraph)
            rpr_camera.set_exposure(scene.camera.data.rpr.motion_blur_exposure)
            self.set_motion_blur_mode(scene)

        # adaptive subdivision will be limited to the current scene render size
        self.rpr_context.enable_aov(pyrpr.AOV_COLOR)
        self.rpr_context.sync_auto_adapt_subdivision()

        self.rpr_context.sync_portal_lights()

        # Exported scene will be rendered vertically flipped, flip it back
        self.rpr_context.set_parameter(pyrpr.CONTEXT_Y_FLIP, True)

        log('Finish sync')


    @staticmethod
    def collect_keyframes(scene):
        """
        Get all the unique keyframes for every animated object, including all the invisible in case visibility is animated
        @return: keyframe times per category(location, rotation, scale) per scene object
        @rtype: dict
        """
        keyframes = {}

        start_frame = scene.frame_start
        end_frame = scene.frame_end

        for blender_object in scene.objects:
            # object has no animation?
            if not blender_object.animation_data or not blender_object.animation_data.action:
                continue

            obj_keyframes = set()

            # animation matrix is exported as a whole, combine all curves keys
            for curve in blender_object.animation_data.action.fcurves.values():
                for key in curve.keyframe_points.values():
                    # Ignore interpolation type, only LINEAR is supported right now
                    # Ignore value as well, collect time only
                    key_time = min(end_frame, max(start_frame, key.co[0]))
                    obj_keyframes.add(key_time)

            # sort collected key times to simplify debug
            obj_keyframes = tuple(sorted(obj_keyframes))

            keyframes[blender_object.name] = obj_keyframes

        return keyframes

    def collect_animation_data_by_keyframes(self, context, keyframes):
        animation_data = {}

        current_frame = context.scene.frame_current

        for name, keyframes in keyframes.items():
            combined = {'translation': [], 'rotation': [], 'scale': []}

            for frame in keyframes:
                context.scene.frame_set(frame)
                depsgraph = context.evaluated_depsgraph_get()
                matrix = self.get_local_matrix(depsgraph.objects[name])

                combined['translation'].append(tuple(matrix.to_translation()))
                quaternion = matrix.to_quaternion()
                combined['rotation'].append([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
                combined['scale'].append(tuple(matrix.to_scale()))

            animation_data[name] = combined

            context.scene.frame_set(current_frame)

        return animation_data

    def get_local_matrix(self, obj):
        """ Get an object local matrix with coordinate axis conversion if needed """

        # TODO check if conversion needed for the NorthStar usage
        #if not obj.parent:  # parent matrix is already adjusted
        #    return self.global_matrix @ obj.matrix_local
        return obj.matrix_local

    @staticmethod
    def get_group_names(obj):
        parent_group_name = "Root"
        group_name = f"Root.{obj.name}"
        if obj.parent:
            if obj.parent.parent:
                parent_group_name = f"{obj.parent.parent.name}.{obj.parent.name}"
            else:
                parent_group_name = f"Root.{obj.parent.name}"
            group_name = f"{obj.parent.name}.{obj.name}"

        return group_name, parent_group_name

    @staticmethod
    def group_transform_from_matrix(matrix):
        """
        Convert matrix to list of 10 floats - transformation(3), rotation quaternion(4), scale(3)
        """
        transform = list(matrix.to_translation())
        quaternion = matrix.to_quaternion()
        transform.extend(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])  # Blender (w,x,y,z) -> GLTF (x,y,z,w)
        transform.extend(matrix.to_scale())
        transform = np.array(transform, dtype=np.float32)
        transform_data = pyrpr.ffi.cast('float*', transform.ctypes.data)
        return transform, transform_data

    @staticmethod
    def assing_group_to_object(rpr_obj, group_name):
        if isinstance(rpr_obj, pyrpr.Shape):
            pyrpr_load_store.rprs_assign_shape_to_group(rpr_obj, group_name)
        elif isinstance(rpr_obj, pyrpr.Light):
            pyrpr_load_store.rprs_assign_light_to_group(rpr_obj, group_name)
        elif isinstance(rpr_obj, pyrpr.Camera):
            pyrpr_load_store.rprs_assign_camera_to_group(rpr_obj, group_name)

    @staticmethod
    def apply_object_animation(keyframes, animation_data, object_name, group_name):
        """ Apply object_name object animation data to group_name """
        data = animation_data.get(object_name, None)
        if not data:
            return

        keys = keyframes.get(object_name)
        for category in ('translation', 'rotation', 'scale'):
            animation = pyrpr_load_store.Animation(group_name, category, keys, data[category])

            pyrpr_load_store.rprs_apply_animation(animation)

    def sync_animated_objects(self, depsgraph, with_camera):
        identity = np.identity(4, dtype=np.float32)
        for obj in self.depsgraph_objects(depsgraph, with_camera=with_camera):
            indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)
            rpr_obj = object.sync(self.rpr_context, obj, indirect_only=indirect_only)
            if not rpr_obj:
                continue

            rpr_obj.set_name(obj.name)
            rpr_obj.set_transform(identity)

            group_name, parent_group_name = self.get_group_names(obj)

            self.assing_group_to_object(rpr_obj, group_name)
            if obj.parent:
                pyrpr_load_store.rprs_assign_parent_group_to_group(group_name, parent_group_name)

            pyrpr_load_store.rprs_set_transform_to_group(group_name, self.get_local_matrix(obj))

