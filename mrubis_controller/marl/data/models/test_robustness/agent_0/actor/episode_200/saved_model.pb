??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
x
layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*
shared_namelayer_0/kernel
q
"layer_0/kernel/Read/ReadVariableOpReadVariableOplayer_0/kernel*
_output_shapes

:$*
dtype0
p
layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namelayer_0/bias
i
 layer_0/bias/Read/ReadVariableOpReadVariableOplayer_0/bias*
_output_shapes
:$*
dtype0
~
layer_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$H*"
shared_namelayer_last/kernel
w
%layer_last/kernel/Read/ReadVariableOpReadVariableOplayer_last/kernel*
_output_shapes

:$H*
dtype0
v
layer_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H* 
shared_namelayer_last/bias
o
#layer_last/bias/Read/ReadVariableOpReadVariableOplayer_last/bias*
_output_shapes
:H*
dtype0
t
probs/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*
shared_nameprobs/kernel
m
 probs/kernel/Read/ReadVariableOpReadVariableOpprobs/kernel*
_output_shapes

:H*
dtype0
l

probs/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
probs/bias
e
probs/bias/Read/ReadVariableOpReadVariableOp
probs/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
?
non_trainable_variables
layer_regularization_losses
	variables
layer_metrics

 layers
trainable_variables
!metrics
regularization_losses
 
ZX
VARIABLE_VALUElayer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
"non_trainable_variables
#layer_regularization_losses
	variables
$layer_metrics

%layers
regularization_losses
trainable_variables
&metrics
][
VARIABLE_VALUElayer_last/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer_last/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
'non_trainable_variables
(layer_regularization_losses
	variables
)layer_metrics

*layers
regularization_losses
trainable_variables
+metrics
XV
VARIABLE_VALUEprobs/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
probs/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
,non_trainable_variables
-layer_regularization_losses
	variables
.layer_metrics

/layers
regularization_losses
trainable_variables
0metrics
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
serving_default_deltaPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_deltaserving_default_inputlayer_0/kernellayer_0/biaslayer_last/kernellayer_last/biasprobs/kernel
probs/bias*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_396353
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer_0/kernel/Read/ReadVariableOp layer_0/bias/Read/ReadVariableOp%layer_last/kernel/Read/ReadVariableOp#layer_last/bias/Read/ReadVariableOp probs/kernel/Read/ReadVariableOpprobs/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_396547
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_0/kernellayer_0/biaslayer_last/kernellayer_last/biasprobs/kernel
probs/bias*
Tin
	2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_396577??
?
{
&__inference_probs_layer_call_fn_396501

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_probs_layer_call_and_return_conditional_losses_3962192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
!__inference__wrapped_model_396149	
input	
delta0
,model_layer_0_matmul_readvariableop_resource1
-model_layer_0_biasadd_readvariableop_resource3
/model_layer_last_matmul_readvariableop_resource4
0model_layer_last_biasadd_readvariableop_resource.
*model_probs_matmul_readvariableop_resource/
+model_probs_biasadd_readvariableop_resource
identity??
#model/layer_0/MatMul/ReadVariableOpReadVariableOp,model_layer_0_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02%
#model/layer_0/MatMul/ReadVariableOp?
model/layer_0/MatMulMatMulinput+model/layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/layer_0/MatMul?
$model/layer_0/BiasAdd/ReadVariableOpReadVariableOp-model_layer_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$model/layer_0/BiasAdd/ReadVariableOp?
model/layer_0/BiasAddBiasAddmodel/layer_0/MatMul:product:0,model/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model/layer_0/BiasAdd?
model/layer_0/ReluRelumodel/layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
model/layer_0/Relu?
&model/layer_last/MatMul/ReadVariableOpReadVariableOp/model_layer_last_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02(
&model/layer_last/MatMul/ReadVariableOp?
model/layer_last/MatMulMatMul model/layer_0/Relu:activations:0.model/layer_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/layer_last/MatMul?
'model/layer_last/BiasAdd/ReadVariableOpReadVariableOp0model_layer_last_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'model/layer_last/BiasAdd/ReadVariableOp?
model/layer_last/BiasAddBiasAdd!model/layer_last/MatMul:product:0/model/layer_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/layer_last/BiasAdd?
model/layer_last/ReluRelu!model/layer_last/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
model/layer_last/Relu?
!model/probs/MatMul/ReadVariableOpReadVariableOp*model_probs_matmul_readvariableop_resource*
_output_shapes

:H*
dtype02#
!model/probs/MatMul/ReadVariableOp?
model/probs/MatMulMatMul#model/layer_last/Relu:activations:0)model/probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/probs/MatMul?
"model/probs/BiasAdd/ReadVariableOpReadVariableOp+model_probs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/probs/BiasAdd/ReadVariableOp?
model/probs/BiasAddBiasAddmodel/probs/MatMul:product:0*model/probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/probs/BiasAdd?
model/probs/SoftmaxSoftmaxmodel/probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/probs/Softmaxq
IdentityIdentitymodel/probs/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::N J
'
_output_shapes
:?????????

_user_specified_nameinput:NJ
'
_output_shapes
:?????????

_user_specified_namedelta:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_model_layer_call_fn_396333	
input	
delta
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3963182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput:NJ
'
_output_shapes
:?????????

_user_specified_namedelta:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?#
?
__inference__traced_save_396547
file_prefix-
)savev2_layer_0_kernel_read_readvariableop+
'savev2_layer_0_bias_read_readvariableop0
,savev2_layer_last_kernel_read_readvariableop.
*savev2_layer_last_bias_read_readvariableop+
'savev2_probs_kernel_read_readvariableop)
%savev2_probs_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_503f09f5056a40d7be8f4daa169fe0d2/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_0_kernel_read_readvariableop'savev2_layer_0_bias_read_readvariableop,savev2_layer_last_kernel_read_readvariableop*savev2_layer_last_bias_read_readvariableop'savev2_probs_kernel_read_readvariableop%savev2_probs_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: :$:$:$H:H:H:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:$: 

_output_shapes
:$:$ 

_output_shapes

:$H: 

_output_shapes
:H:$ 

_output_shapes

:H: 

_output_shapes
::

_output_shapes
: 
?
?
C__inference_layer_0_layer_call_and_return_conditional_losses_396165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_probs_layer_call_and_return_conditional_losses_396219

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H:::O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_layer_0_layer_call_and_return_conditional_losses_396452

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_model_layer_call_fn_396423
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3962802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_probs_layer_call_and_return_conditional_losses_396492

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????H:::O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_model_layer_call_fn_396295	
input	
delta
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3962802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput:NJ
'
_output_shapes
:?????????

_user_specified_namedelta:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396318

inputs
inputs_1
layer_0_396302
layer_0_396304
layer_last_396307
layer_last_396309
probs_396312
probs_396314
identity??layer_0/StatefulPartitionedCall?"layer_last/StatefulPartitionedCall?probs/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_396302layer_0_396304*
Tin
2*
Tout
2*'
_output_shapes
:?????????$*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_3961652!
layer_0/StatefulPartitionedCall?
"layer_last/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_last_396307layer_last_396309*
Tin
2*
Tout
2*'
_output_shapes
:?????????H*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_layer_last_layer_call_and_return_conditional_losses_3961922$
"layer_last/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall+layer_last/StatefulPartitionedCall:output:0probs_396312probs_396314*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_probs_layer_call_and_return_conditional_losses_3962192
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall#^layer_last/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2H
"layer_last/StatefulPartitionedCall"layer_last/StatefulPartitionedCall2>
probs/StatefulPartitionedCallprobs/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396405
inputs_0
inputs_1*
&layer_0_matmul_readvariableop_resource+
'layer_0_biasadd_readvariableop_resource-
)layer_last_matmul_readvariableop_resource.
*layer_last_biasadd_readvariableop_resource(
$probs_matmul_readvariableop_resource)
%probs_biasadd_readvariableop_resource
identity??
layer_0/MatMul/ReadVariableOpReadVariableOp&layer_0_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
layer_0/MatMul/ReadVariableOp?
layer_0/MatMulMatMulinputs_0%layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
layer_0/MatMul?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAddlayer_0/MatMul:product:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
layer_0/BiasAddp
layer_0/ReluRelulayer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
layer_0/Relu?
 layer_last/MatMul/ReadVariableOpReadVariableOp)layer_last_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02"
 layer_last/MatMul/ReadVariableOp?
layer_last/MatMulMatMullayer_0/Relu:activations:0(layer_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
layer_last/MatMul?
!layer_last/BiasAdd/ReadVariableOpReadVariableOp*layer_last_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02#
!layer_last/BiasAdd/ReadVariableOp?
layer_last/BiasAddBiasAddlayer_last/MatMul:product:0)layer_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
layer_last/BiasAddy
layer_last/ReluRelulayer_last/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
layer_last/Relu?
probs/MatMul/ReadVariableOpReadVariableOp$probs_matmul_readvariableop_resource*
_output_shapes

:H*
dtype02
probs/MatMul/ReadVariableOp?
probs/MatMulMatMullayer_last/Relu:activations:0#probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
probs/MatMul?
probs/BiasAdd/ReadVariableOpReadVariableOp%probs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
probs/BiasAdd/ReadVariableOp?
probs/BiasAddBiasAddprobs/MatMul:product:0$probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
probs/BiasAdds
probs/SoftmaxSoftmaxprobs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
probs/Softmaxk
IdentityIdentityprobs/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
$__inference_signature_wrapper_396353	
delta	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_3961492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namedelta:NJ
'
_output_shapes
:?????????

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_layer_last_layer_call_fn_396481

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????H*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_layer_last_layer_call_and_return_conditional_losses_3961922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?"
?
"__inference__traced_restore_396577
file_prefix#
assignvariableop_layer_0_kernel#
assignvariableop_1_layer_0_bias(
$assignvariableop_2_layer_last_kernel&
"assignvariableop_3_layer_last_bias#
assignvariableop_4_probs_kernel!
assignvariableop_5_probs_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer_0_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_0_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_layer_last_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_layer_last_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_probs_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_probs_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_layer_last_layer_call_and_return_conditional_losses_396192

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$:::O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396379
inputs_0
inputs_1*
&layer_0_matmul_readvariableop_resource+
'layer_0_biasadd_readvariableop_resource-
)layer_last_matmul_readvariableop_resource.
*layer_last_biasadd_readvariableop_resource(
$probs_matmul_readvariableop_resource)
%probs_biasadd_readvariableop_resource
identity??
layer_0/MatMul/ReadVariableOpReadVariableOp&layer_0_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
layer_0/MatMul/ReadVariableOp?
layer_0/MatMulMatMulinputs_0%layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
layer_0/MatMul?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAddlayer_0/MatMul:product:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
layer_0/BiasAddp
layer_0/ReluRelulayer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
layer_0/Relu?
 layer_last/MatMul/ReadVariableOpReadVariableOp)layer_last_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02"
 layer_last/MatMul/ReadVariableOp?
layer_last/MatMulMatMullayer_0/Relu:activations:0(layer_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
layer_last/MatMul?
!layer_last/BiasAdd/ReadVariableOpReadVariableOp*layer_last_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02#
!layer_last/BiasAdd/ReadVariableOp?
layer_last/BiasAddBiasAddlayer_last/MatMul:product:0)layer_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
layer_last/BiasAddy
layer_last/ReluRelulayer_last/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
layer_last/Relu?
probs/MatMul/ReadVariableOpReadVariableOp$probs_matmul_readvariableop_resource*
_output_shapes

:H*
dtype02
probs/MatMul/ReadVariableOp?
probs/MatMulMatMullayer_last/Relu:activations:0#probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
probs/MatMul?
probs/BiasAdd/ReadVariableOpReadVariableOp%probs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
probs/BiasAdd/ReadVariableOp?
probs/BiasAddBiasAddprobs/MatMul:product:0$probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
probs/BiasAdds
probs/SoftmaxSoftmaxprobs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
probs/Softmaxk
IdentityIdentityprobs/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396280

inputs
inputs_1
layer_0_396264
layer_0_396266
layer_last_396269
layer_last_396271
probs_396274
probs_396276
identity??layer_0/StatefulPartitionedCall?"layer_last/StatefulPartitionedCall?probs/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_396264layer_0_396266*
Tin
2*
Tout
2*'
_output_shapes
:?????????$*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_3961652!
layer_0/StatefulPartitionedCall?
"layer_last/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_last_396269layer_last_396271*
Tin
2*
Tout
2*'
_output_shapes
:?????????H*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_layer_last_layer_call_and_return_conditional_losses_3961922$
"layer_last/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall+layer_last/StatefulPartitionedCall:output:0probs_396274probs_396276*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_probs_layer_call_and_return_conditional_losses_3962192
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall#^layer_last/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2H
"layer_last/StatefulPartitionedCall"layer_last/StatefulPartitionedCall2>
probs/StatefulPartitionedCallprobs/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396256	
input	
delta
layer_0_396240
layer_0_396242
layer_last_396245
layer_last_396247
probs_396250
probs_396252
identity??layer_0/StatefulPartitionedCall?"layer_last/StatefulPartitionedCall?probs/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputlayer_0_396240layer_0_396242*
Tin
2*
Tout
2*'
_output_shapes
:?????????$*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_3961652!
layer_0/StatefulPartitionedCall?
"layer_last/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_last_396245layer_last_396247*
Tin
2*
Tout
2*'
_output_shapes
:?????????H*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_layer_last_layer_call_and_return_conditional_losses_3961922$
"layer_last/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall+layer_last/StatefulPartitionedCall:output:0probs_396250probs_396252*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_probs_layer_call_and_return_conditional_losses_3962192
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall#^layer_last/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2H
"layer_last/StatefulPartitionedCall"layer_last/StatefulPartitionedCall2>
probs/StatefulPartitionedCallprobs/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput:NJ
'
_output_shapes
:?????????

_user_specified_namedelta:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_396236	
input	
delta
layer_0_396176
layer_0_396178
layer_last_396203
layer_last_396205
probs_396230
probs_396232
identity??layer_0/StatefulPartitionedCall?"layer_last/StatefulPartitionedCall?probs/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputlayer_0_396176layer_0_396178*
Tin
2*
Tout
2*'
_output_shapes
:?????????$*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_3961652!
layer_0/StatefulPartitionedCall?
"layer_last/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_last_396203layer_last_396205*
Tin
2*
Tout
2*'
_output_shapes
:?????????H*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_layer_last_layer_call_and_return_conditional_losses_3961922$
"layer_last/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall+layer_last/StatefulPartitionedCall:output:0probs_396230probs_396232*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_probs_layer_call_and_return_conditional_losses_3962192
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall#^layer_last/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2H
"layer_last/StatefulPartitionedCall"layer_last/StatefulPartitionedCall2>
probs/StatefulPartitionedCallprobs/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput:NJ
'
_output_shapes
:?????????

_user_specified_namedelta:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_model_layer_call_fn_396441
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3963182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_layer_0_layer_call_fn_396461

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????$*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_3961652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_layer_last_layer_call_and_return_conditional_losses_396472

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$:::O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
delta.
serving_default_delta:0?????????
7
input.
serving_default_input:0?????????9
probs0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ʁ
?%
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
1_default_save_signature
*2&call_and_return_all_conditional_losses
3__call__"?#
_tf_keras_model?"{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer_last", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_last", "inbound_nodes": [[["layer_0", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}, "name": "delta", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "probs", "inbound_nodes": [[["layer_last", 0, 0, {}]]]}], "input_layers": [["input", 0, 0], ["delta", 0, 0]], "output_layers": [["probs", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer_last", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_last", "inbound_nodes": [[["layer_0", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}, "name": "delta", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "probs", "inbound_nodes": [[["layer_last", 0, 0, {}]]]}], "input_layers": [["input", 0, 0], ["delta", 0, 0]], "output_layers": [["probs", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*4&call_and_return_all_conditional_losses
5__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*6&call_and_return_all_conditional_losses
7__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_last", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer_last", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "delta", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*8&call_and_return_all_conditional_losses
9__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
layer_regularization_losses
	variables
layer_metrics

 layers
trainable_variables
!metrics
regularization_losses
3__call__
1_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
,
:serving_default"
signature_map
 :$2layer_0/kernel
:$2layer_0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
"non_trainable_variables
#layer_regularization_losses
	variables
$layer_metrics

%layers
regularization_losses
trainable_variables
&metrics
5__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
#:!$H2layer_last/kernel
:H2layer_last/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
'non_trainable_variables
(layer_regularization_losses
	variables
)layer_metrics

*layers
regularization_losses
trainable_variables
+metrics
7__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
:H2probs/kernel
:2
probs/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
,non_trainable_variables
-layer_regularization_losses
	variables
.layer_metrics

/layers
regularization_losses
trainable_variables
0metrics
9__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
!__inference__wrapped_model_396149?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *J?G
E?B
?
input?????????
?
delta?????????
?2?
A__inference_model_layer_call_and_return_conditional_losses_396405
A__inference_model_layer_call_and_return_conditional_losses_396379
A__inference_model_layer_call_and_return_conditional_losses_396256
A__inference_model_layer_call_and_return_conditional_losses_396236?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_model_layer_call_fn_396423
&__inference_model_layer_call_fn_396295
&__inference_model_layer_call_fn_396333
&__inference_model_layer_call_fn_396441?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_layer_0_layer_call_and_return_conditional_losses_396452?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_layer_0_layer_call_fn_396461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_layer_last_layer_call_and_return_conditional_losses_396472?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_layer_last_layer_call_fn_396481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_probs_layer_call_and_return_conditional_losses_396492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_probs_layer_call_fn_396501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
6B4
$__inference_signature_wrapper_396353deltainput?
!__inference__wrapped_model_396149?T?Q
J?G
E?B
?
input?????????
?
delta?????????
? "-?*
(
probs?
probs??????????
C__inference_layer_0_layer_call_and_return_conditional_losses_396452\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????$
? {
(__inference_layer_0_layer_call_fn_396461O/?,
%?"
 ?
inputs?????????
? "??????????$?
F__inference_layer_last_layer_call_and_return_conditional_losses_396472\/?,
%?"
 ?
inputs?????????$
? "%?"
?
0?????????H
? ~
+__inference_layer_last_layer_call_fn_396481O/?,
%?"
 ?
inputs?????????$
? "??????????H?
A__inference_model_layer_call_and_return_conditional_losses_396236?\?Y
R?O
E?B
?
input?????????
?
delta?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_396256?\?Y
R?O
E?B
?
input?????????
?
delta?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_396379?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_396405?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_396295?\?Y
R?O
E?B
?
input?????????
?
delta?????????
p

 
? "???????????
&__inference_model_layer_call_fn_396333?\?Y
R?O
E?B
?
input?????????
?
delta?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_396423?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_396441?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
A__inference_probs_layer_call_and_return_conditional_losses_396492\/?,
%?"
 ?
inputs?????????H
? "%?"
?
0?????????
? y
&__inference_probs_layer_call_fn_396501O/?,
%?"
 ?
inputs?????????H
? "???????????
$__inference_signature_wrapper_396353?a?^
? 
W?T
(
delta?
delta?????????
(
input?
input?????????"-?*
(
probs?
probs?????????