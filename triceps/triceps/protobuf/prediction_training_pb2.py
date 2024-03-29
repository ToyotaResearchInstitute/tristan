# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: prediction_training.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x19prediction_training.proto\x12\tdgp.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x19google/protobuf/any.proto"\xa9\x02\n\x10TimestampedState\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12$\n\x08position\x18\x02 \x01(\x0b\x32\x12.dgp.proto.Point3D\x12\x10\n\x08velocity\x18\x03 \x03(\x02\x12\x13\n\x0brotation_aw\x18\x04 \x03(\x02\x12\x0f\n\x07heading\x18\x05 \x03(\x02\x12\r\n\x05width\x18\x06 \x01(\x02\x12\x0e\n\x06length\x18\x07 \x01(\x02\x12\x0e\n\x06height\x18\x08 \x01(\x02\x12\x44\n\x14uncertainty_estimate\x18\t \x01(\x0b\x32&.dgp.proto.TimestampedStateUncertainty\x12\x13\n\x0bobject_size\x18\n \x01(\x02"\x89\x01\n\x1bTimestampedStateUncertainty\x12\x34\n\x13position_covariance\x18\x01 \x01(\x0b\x32\x17.dgp.proto.Covariance3D\x12\x34\n\x13velocity_covariance\x18\x02 \x01(\x0b\x32\x17.dgp.proto.Covariance3D";\n\x0cRawImageData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05width\x18\x02 \x01(\x03\x12\x0e\n\x06height\x18\x03 \x01(\x03"\xe4\x01\n\x0bSensorImage\x12\x12\n\x08\x66ilename\x18\x01 \x01(\tH\x00\x12+\n\x08raw_data\x18\x03 \x01(\x0b\x32\x17.dgp.proto.RawImageDataH\x00\x12\x14\n\x0c\x64gp_datum_id\x18\x02 \x01(\t\x12<\n\x0cimage_format\x18\x04 \x01(\x0e\x32&.dgp.proto.SensorImage.ImageFormatEnum"1\n\x0fImageFormatEnum\x12\x0b\n\x07Unknown\x10\x00\x12\x07\n\x03\x42gr\x10\x01\x12\x08\n\x04Jpeg\x10\x02\x42\r\n\x0b\x64\x61ta_source"\xae\x02\n\x1aTimestampedPredictionInput\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x31\n\rtimestamp_end\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\x12sensor_image_input\x18\x03 \x01(\x0b\x32\x16.dgp.proto.SensorImage\x12\x1e\n\x16sensor_image_sensor_id\x18\x04 \x01(\t\x12\x14\n\x0cvector_input\x18\x05 \x03(\x02\x12\x1c\n\x14vector_input_type_id\x18\x06 \x01(\t\x12&\n\tlocal_map\x18\x07 \x01(\x0b\x32\x13.dgp.proto.LocalMap"\xb3\x02\n\x11TrafficLightState\x12\x10\n\x08state_id\x18\x01 \x01(\t\x12\x16\n\x0emap_element_id\x18\x02 \x01(\t\x12\r\n\x05state\x18\x03 \x01(\t\x12\x33\n\x0ftimestamp_start\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x31\n\rtimestamp_end\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12&\n\nstop_point\x18\x06 \x03(\x0b\x32\x12.dgp.proto.Point3D\x12\x1a\n\x12information_source\x18\x07 \x01(\t\x12\x1a\n\x12traffic_light_type\x18\x08 \x01(\t\x12\x1d\n\x15\x61\x64\x64itional_state_info\x18\t \x01(\t"\xca\x01\n\x19TimestampedSemanticTarget\x12\x33\n\x0ftimestamp_start\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x31\n\rtimestamp_end\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x13\n\x0bis_interval\x18\x03 \x01(\x08\x12\x10\n\x08\x61gent_id\x18\x04 \x01(\t\x12\x0f\n\x07type_id\x18\x05 \x01(\t\x12\r\n\x05value\x18\x06 \x01(\t"\x85\x01\n\rMapConnection\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x13\n\x0bstart_index\x18\x03 \x01(\x05\x12\x11\n\tend_index\x18\x04 \x01(\x05\x12\x19\n\x11other_start_index\x18\x05 \x01(\x05\x12\x17\n\x0fother_end_index\x18\x06 \x01(\x05"\xfa\x01\n\x04Lane\x12\n\n\x02id\x18\x01 \x01(\t\x12\'\n\x0b\x63\x65nter_line\x18\x02 \x03(\x0b\x32\x12.dgp.proto.Segment\x12.\n\x12left_lane_boundary\x18\x03 \x03(\x0b\x32\x12.dgp.proto.Segment\x12/\n\x13right_lane_boundary\x18\x04 \x03(\x0b\x32\x12.dgp.proto.Segment\x12)\n\rlane_boundary\x18\x05 \x03(\x0b\x32\x12.dgp.proto.Segment\x12\x31\n\x0fmap_connections\x18\x06 \x03(\x0b\x32\x18.dgp.proto.MapConnection"\xb1\x01\n\nMapFeature\x12\n\n\x02id\x18\x01 \x01(\t\x12$\n\x08segments\x18\x02 \x03(\x0b\x32\x12.dgp.proto.Segment\x12\x1c\n\x0fspeed_limit_mph\x18\x03 \x01(\x02H\x00\x88\x01\x01\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x31\n\x0fmap_connections\x18\x05 \x03(\x0b\x32\x18.dgp.proto.MapConnectionB\x12\n\x10_speed_limit_mph"\xb5\x01\n\x0f\x41gentTrajectory\x12\x10\n\x08\x61gent_id\x18\x01 \x01(\t\x12/\n\ntrajectory\x18\x02 \x03(\x0b\x32\x1b.dgp.proto.TimestampedState\x12@\n\x11\x61\x64\x64itional_inputs\x18\x03 \x03(\x0b\x32%.dgp.proto.TimestampedPredictionInput\x12\x1d\n\x15\x61\x64\x64itional_agent_info\x18\x04 \x01(\t"*\n\x07Point3D\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02"V\n\x0c\x43ovariance3D\x12\n\n\x02xx\x18\x01 \x01(\x02\x12\n\n\x02xy\x18\x02 \x01(\x02\x12\n\n\x02xz\x18\x03 \x01(\x02\x12\n\n\x02yy\x18\x04 \x01(\x02\x12\n\n\x02yz\x18\x05 \x01(\x02\x12\n\n\x02zz\x18\x06 \x01(\x02"M\n\x07Segment\x12!\n\x05start\x18\x01 \x01(\x0b\x32\x12.dgp.proto.Point3D\x12\x1f\n\x03\x65nd\x18\x02 \x01(\x0b\x32\x12.dgp.proto.Point3D"E\n\x04Zone\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12#\n\x07polygon\x18\x03 \x03(\x0b\x32\x12.dgp.proto.Segment"\xce\x01\n\x08LocalMap\x12\x1e\n\x05lanes\x18\x01 \x03(\x0b\x32\x0f.dgp.proto.Lane\x12\x1e\n\x05zones\x18\x02 \x03(\x0b\x32\x0f.dgp.proto.Zone\x12\x17\n\x0f\x65xternal_map_id\x18\x03 \x01(\t\x12&\n\x08map_data\x18\x04 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x14\n\x0cmap_metainfo\x18\x05 \x01(\t\x12+\n\x0cmap_features\x18\x06 \x03(\x0b\x32\x15.dgp.proto.MapFeature"\xe2\x04\n\x12PredictionInstance\x12\x13\n\x0binstance_id\x18\x01 \x01(\t\x12P\n\x12\x61gent_trajectories\x18\x02 \x03(\x0b\x32\x34.dgp.proto.PredictionInstance.AgentTrajectoriesEntry\x12,\n\x0fmap_information\x18\x03 \x01(\x0b\x32\x13.dgp.proto.LocalMap\x12>\n\x10semantic_targets\x18\x04 \x03(\x0b\x32$.dgp.proto.TimestampedSemanticTarget\x12\x15\n\regovehicle_id\x18\x05 \x01(\t\x12\x33\n\x0fprediction_time\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12 \n\x18prediction_instance_info\x18\x07 \x01(\t\x12\x11\n\tis_leaked\x18\x08 \x01(\x08\x12:\n\x16source_prediction_time\x18\t \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12$\n\x1cprediction_instance_pii_info\x18\n \x01(\t\x12>\n\x15\x61\x64\x64itional_scene_info\x18\x0b \x03(\x0b\x32\x1f.dgp.proto.AdditionalSceneDatum\x1aT\n\x16\x41gentTrajectoriesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.dgp.proto.AgentTrajectory:\x02\x38\x01"\x81\x01\n\x14\x41\x64\x64itionalSceneDatum\x12\x1d\n\x13general_scene_datum\x18\x01 \x01(\tH\x00\x12<\n\x14traffic_light_states\x18\x02 \x01(\x0b\x32\x1c.dgp.proto.TrafficLightStateH\x00\x42\x0c\n\nDatumUnion"p\n\x18PredictionSetInformation\x12\x13\n\x0b\x64\x61ta_source\x18\x01 \x01(\t\x12\x31\n\rcreation_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04\x63ity\x18\x03 \x01(\t"\xa0\x01\n\rPredictionSet\x12;\n\x14prediction_instances\x18\x01 \x03(\x0b\x32\x1d.dgp.proto.PredictionInstance\x12\x38\n\x0binformation\x18\x02 \x01(\x0b\x32#.dgp.proto.PredictionSetInformation\x12\x18\n\x10instance_weights\x18\x03 \x03(\x02\x32[\n\x13RadDrivingInterface\x12\x44\n\x0cGetInference\x12\x18.dgp.proto.PredictionSet\x1a\x18.dgp.proto.PredictionSet"\x00\x62\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "prediction_training_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _PREDICTIONINSTANCE_AGENTTRAJECTORIESENTRY._options = None
    _PREDICTIONINSTANCE_AGENTTRAJECTORIESENTRY._serialized_options = b"8\001"
    _TIMESTAMPEDSTATE._serialized_start = 101
    _TIMESTAMPEDSTATE._serialized_end = 398
    _TIMESTAMPEDSTATEUNCERTAINTY._serialized_start = 401
    _TIMESTAMPEDSTATEUNCERTAINTY._serialized_end = 538
    _RAWIMAGEDATA._serialized_start = 540
    _RAWIMAGEDATA._serialized_end = 599
    _SENSORIMAGE._serialized_start = 602
    _SENSORIMAGE._serialized_end = 830
    _SENSORIMAGE_IMAGEFORMATENUM._serialized_start = 766
    _SENSORIMAGE_IMAGEFORMATENUM._serialized_end = 815
    _TIMESTAMPEDPREDICTIONINPUT._serialized_start = 833
    _TIMESTAMPEDPREDICTIONINPUT._serialized_end = 1135
    _TRAFFICLIGHTSTATE._serialized_start = 1138
    _TRAFFICLIGHTSTATE._serialized_end = 1445
    _TIMESTAMPEDSEMANTICTARGET._serialized_start = 1448
    _TIMESTAMPEDSEMANTICTARGET._serialized_end = 1650
    _MAPCONNECTION._serialized_start = 1653
    _MAPCONNECTION._serialized_end = 1786
    _LANE._serialized_start = 1789
    _LANE._serialized_end = 2039
    _MAPFEATURE._serialized_start = 2042
    _MAPFEATURE._serialized_end = 2219
    _AGENTTRAJECTORY._serialized_start = 2222
    _AGENTTRAJECTORY._serialized_end = 2403
    _POINT3D._serialized_start = 2405
    _POINT3D._serialized_end = 2447
    _COVARIANCE3D._serialized_start = 2449
    _COVARIANCE3D._serialized_end = 2535
    _SEGMENT._serialized_start = 2537
    _SEGMENT._serialized_end = 2614
    _ZONE._serialized_start = 2616
    _ZONE._serialized_end = 2685
    _LOCALMAP._serialized_start = 2688
    _LOCALMAP._serialized_end = 2894
    _PREDICTIONINSTANCE._serialized_start = 2897
    _PREDICTIONINSTANCE._serialized_end = 3507
    _PREDICTIONINSTANCE_AGENTTRAJECTORIESENTRY._serialized_start = 3423
    _PREDICTIONINSTANCE_AGENTTRAJECTORIESENTRY._serialized_end = 3507
    _ADDITIONALSCENEDATUM._serialized_start = 3510
    _ADDITIONALSCENEDATUM._serialized_end = 3639
    _PREDICTIONSETINFORMATION._serialized_start = 3641
    _PREDICTIONSETINFORMATION._serialized_end = 3753
    _PREDICTIONSET._serialized_start = 3756
    _PREDICTIONSET._serialized_end = 3916
    _RADDRIVINGINTERFACE._serialized_start = 3918
    _RADDRIVINGINTERFACE._serialized_end = 4009
# @@protoc_insertion_point(module_scope)
