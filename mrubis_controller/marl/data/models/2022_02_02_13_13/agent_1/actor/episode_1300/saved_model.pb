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
z
dense1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$* 
shared_namedense1_1/kernel
s
#dense1_1/kernel/Read/ReadVariableOpReadVariableOpdense1_1/kernel*
_output_shapes

:$*
dtype0
r
dense1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense1_1/bias
k
!dense1_1/bias/Read/ReadVariableOpReadVariableOpdense1_1/bias*
_output_shapes
:$*
dtype0
z
dense2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$H* 
shared_namedense2_1/kernel
s
#dense2_1/kernel/Read/ReadVariableOpReadVariableOpdense2_1/kernel*
_output_shapes

:$H*
dtype0
r
dense2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense2_1/bias
k
!dense2_1/bias/Read/ReadVariableOpReadVariableOpdense2_1/bias*
_output_shapes
:H*
dtype0
{
dense3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H?* 
shared_namedense3_1/kernel
t
#dense3_1/kernel/Read/ReadVariableOpReadVariableOpdense3_1/kernel*
_output_shapes
:	H?*
dtype0
s
dense3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense3_1/bias
l
!dense3_1/bias/Read/ReadVariableOpReadVariableOpdense3_1/bias*
_output_shapes	
:?*
dtype0
y
probs_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameprobs_1/kernel
r
"probs_1/kernel/Read/ReadVariableOpReadVariableOpprobs_1/kernel*
_output_shapes
:	?*
dtype0
p
probs_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprobs_1/bias
i
 probs_1/bias/Read/ReadVariableOpReadVariableOpprobs_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
$layer_metrics
trainable_variables
	variables
%metrics
&non_trainable_variables

'layers
	regularization_losses
(layer_regularization_losses
 
[Y
VARIABLE_VALUEdense1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
)layer_metrics
trainable_variables
	variables
*metrics
+non_trainable_variables

,layers
regularization_losses
-layer_regularization_losses
[Y
VARIABLE_VALUEdense2_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense2_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.layer_metrics
trainable_variables
	variables
/metrics
0non_trainable_variables

1layers
regularization_losses
2layer_regularization_losses
[Y
VARIABLE_VALUEdense3_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense3_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3layer_metrics
trainable_variables
	variables
4metrics
5non_trainable_variables

6layers
regularization_losses
7layer_regularization_losses
ZX
VARIABLE_VALUEprobs_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEprobs_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
8layer_metrics
 trainable_variables
!	variables
9metrics
:non_trainable_variables

;layers
"regularization_losses
<layer_regularization_losses
 
 
 
*
0
1
2
3
4
5
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_deltaserving_default_inputdense1_1/kerneldense1_1/biasdense2_1/kerneldense2_1/biasdense3_1/kerneldense3_1/biasprobs_1/kernelprobs_1/bias*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_5817931
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense1_1/kernel/Read/ReadVariableOp!dense1_1/bias/Read/ReadVariableOp#dense2_1/kernel/Read/ReadVariableOp!dense2_1/bias/Read/ReadVariableOp#dense3_1/kernel/Read/ReadVariableOp!dense3_1/bias/Read/ReadVariableOp"probs_1/kernel/Read/ReadVariableOp probs_1/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU 2J 8*)
f$R"
 __inference__traced_save_5818170
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1_1/kerneldense1_1/biasdense2_1/kerneldense2_1/biasdense3_1/kerneldense3_1/biasprobs_1/kernelprobs_1/bias*
Tin
2	*
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
GPU 2J 8*,
f'R%
#__inference__traced_restore_5818206??
?
}
(__inference_dense1_layer_call_fn_5818059

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
C__inference_dense1_layer_call_and_return_conditional_losses_58176902
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
?#
?
"__inference__wrapped_model_5817674	
input	
delta1
-model_3_dense1_matmul_readvariableop_resource2
.model_3_dense1_biasadd_readvariableop_resource1
-model_3_dense2_matmul_readvariableop_resource2
.model_3_dense2_biasadd_readvariableop_resource1
-model_3_dense3_matmul_readvariableop_resource2
.model_3_dense3_biasadd_readvariableop_resource0
,model_3_probs_matmul_readvariableop_resource1
-model_3_probs_biasadd_readvariableop_resource
identity??
$model_3/dense1/MatMul/ReadVariableOpReadVariableOp-model_3_dense1_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02&
$model_3/dense1/MatMul/ReadVariableOp?
model_3/dense1/MatMulMatMulinput,model_3/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model_3/dense1/MatMul?
%model_3/dense1/BiasAdd/ReadVariableOpReadVariableOp.model_3_dense1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02'
%model_3/dense1/BiasAdd/ReadVariableOp?
model_3/dense1/BiasAddBiasAddmodel_3/dense1/MatMul:product:0-model_3/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
model_3/dense1/BiasAdd?
model_3/dense1/ReluRelumodel_3/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
model_3/dense1/Relu?
$model_3/dense2/MatMul/ReadVariableOpReadVariableOp-model_3_dense2_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02&
$model_3/dense2/MatMul/ReadVariableOp?
model_3/dense2/MatMulMatMul!model_3/dense1/Relu:activations:0,model_3/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model_3/dense2/MatMul?
%model_3/dense2/BiasAdd/ReadVariableOpReadVariableOp.model_3_dense2_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02'
%model_3/dense2/BiasAdd/ReadVariableOp?
model_3/dense2/BiasAddBiasAddmodel_3/dense2/MatMul:product:0-model_3/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model_3/dense2/BiasAdd?
model_3/dense2/ReluRelumodel_3/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
model_3/dense2/Relu?
$model_3/dense3/MatMul/ReadVariableOpReadVariableOp-model_3_dense3_matmul_readvariableop_resource*
_output_shapes
:	H?*
dtype02&
$model_3/dense3/MatMul/ReadVariableOp?
model_3/dense3/MatMulMatMul!model_3/dense2/Relu:activations:0,model_3/dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense3/MatMul?
%model_3/dense3/BiasAdd/ReadVariableOpReadVariableOp.model_3_dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model_3/dense3/BiasAdd/ReadVariableOp?
model_3/dense3/BiasAddBiasAddmodel_3/dense3/MatMul:product:0-model_3/dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense3/BiasAdd?
#model_3/probs/MatMul/ReadVariableOpReadVariableOp,model_3_probs_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model_3/probs/MatMul/ReadVariableOp?
model_3/probs/MatMulMatMulmodel_3/dense3/BiasAdd:output:0+model_3/probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/probs/MatMul?
$model_3/probs/BiasAdd/ReadVariableOpReadVariableOp-model_3_probs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model_3/probs/BiasAdd/ReadVariableOp?
model_3/probs/BiasAddBiasAddmodel_3/probs/MatMul:product:0,model_3/probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/probs/BiasAdd?
model_3/probs/SoftmaxSoftmaxmodel_3/probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/probs/Softmaxs
IdentityIdentitymodel_3/probs/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????:::::::::N J
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
C__inference_dense3_layer_call_and_return_conditional_losses_5817743

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

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
?
}
(__inference_dense2_layer_call_fn_5818079

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
GPU 2J 8*L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_58177172
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
?
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817812	
input	
delta
dense1_5817791
dense1_5817793
dense2_5817796
dense2_5817798
dense3_5817801
dense3_5817803
probs_5817806
probs_5817808
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense3/StatefulPartitionedCall?probs/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputdense1_5817791dense1_5817793*
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
C__inference_dense1_layer_call_and_return_conditional_losses_58176902 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_5817796dense2_5817798*
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
GPU 2J 8*L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_58177172 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_5817801dense3_5817803*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_58177432 
dense3/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0probs_5817806probs_5817808*
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
GPU 2J 8*K
fFRD
B__inference_probs_layer_call_and_return_conditional_losses_58177702
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2>
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
: :

_output_shapes
: :	

_output_shapes
: 
?
|
'__inference_probs_layer_call_fn_5818118

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
GPU 2J 8*K
fFRD
B__inference_probs_layer_call_and_return_conditional_losses_58177702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
)__inference_model_3_layer_call_fn_5818017
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_58178412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
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
: :

_output_shapes
: :	

_output_shapes
: 
?

?
)__inference_model_3_layer_call_fn_5817907	
input	
delta
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_58178882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
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
: :

_output_shapes
: :	

_output_shapes
: 
?
}
(__inference_dense3_layer_call_fn_5818098

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_58177432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817841

inputs
inputs_1
dense1_5817820
dense1_5817822
dense2_5817825
dense2_5817827
dense3_5817830
dense3_5817832
probs_5817835
probs_5817837
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense3/StatefulPartitionedCall?probs/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_5817820dense1_5817822*
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
C__inference_dense1_layer_call_and_return_conditional_losses_58176902 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_5817825dense2_5817827*
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
GPU 2J 8*L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_58177172 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_5817830dense3_5817832*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_58177432 
dense3/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0probs_5817835probs_5817837*
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
GPU 2J 8*K
fFRD
B__inference_probs_layer_call_and_return_conditional_losses_58177702
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2>
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
B__inference_probs_layer_call_and_return_conditional_losses_5817770

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?&
?
 __inference__traced_save_5818170
file_prefix.
*savev2_dense1_1_kernel_read_readvariableop,
(savev2_dense1_1_bias_read_readvariableop.
*savev2_dense2_1_kernel_read_readvariableop,
(savev2_dense2_1_bias_read_readvariableop.
*savev2_dense3_1_kernel_read_readvariableop,
(savev2_dense3_1_bias_read_readvariableop-
)savev2_probs_1_kernel_read_readvariableop+
'savev2_probs_1_bias_read_readvariableop
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
value3B1 B+_temp_a4cbe77860c04c4c9e6f2e89975b3527/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense1_1_kernel_read_readvariableop(savev2_dense1_1_bias_read_readvariableop*savev2_dense2_1_kernel_read_readvariableop(savev2_dense2_1_bias_read_readvariableop*savev2_dense3_1_kernel_read_readvariableop(savev2_dense3_1_bias_read_readvariableop)savev2_probs_1_kernel_read_readvariableop'savev2_probs_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
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

identity_1Identity_1:output:0*Z
_input_shapesI
G: :$:$:$H:H:	H?:?:	?:: 2(
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
:H:%!

_output_shapes
:	H?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: 
?
?
C__inference_dense3_layer_call_and_return_conditional_losses_5818089

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

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
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817963
inputs_0
inputs_1)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource(
$probs_matmul_readvariableop_resource)
%probs_biasadd_readvariableop_resource
identity??
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs_0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense2/Relu?
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	H?*
dtype02
dense3/MatMul/ReadVariableOp?
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/MatMul?
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense3/BiasAdd/ReadVariableOp?
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/BiasAdd?
probs/MatMul/ReadVariableOpReadVariableOp$probs_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
probs/MatMul/ReadVariableOp?
probs/MatMulMatMuldense3/BiasAdd:output:0#probs/MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????:::::::::Q M
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
B__inference_probs_layer_call_and_return_conditional_losses_5818109

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
)__inference_model_3_layer_call_fn_5817860	
input	
delta
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_58178412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817995
inputs_0
inputs_1)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource(
$probs_matmul_readvariableop_resource)
%probs_biasadd_readvariableop_resource
identity??
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs_0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:$H*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense2/Relu?
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	H?*
dtype02
dense3/MatMul/ReadVariableOp?
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/MatMul?
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense3/BiasAdd/ReadVariableOp?
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/BiasAdd?
probs/MatMul/ReadVariableOpReadVariableOp$probs_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
probs/MatMul/ReadVariableOp?
probs/MatMulMatMuldense3/BiasAdd:output:0#probs/MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????:::::::::Q M
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817787	
input	
delta
dense1_5817701
dense1_5817703
dense2_5817728
dense2_5817730
dense3_5817754
dense3_5817756
probs_5817781
probs_5817783
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense3/StatefulPartitionedCall?probs/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputdense1_5817701dense1_5817703*
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
C__inference_dense1_layer_call_and_return_conditional_losses_58176902 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_5817728dense2_5817730*
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
GPU 2J 8*L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_58177172 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_5817754dense3_5817756*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_58177432 
dense3/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0probs_5817781probs_5817783*
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
GPU 2J 8*K
fFRD
B__inference_probs_layer_call_and_return_conditional_losses_58177702
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2>
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
: :

_output_shapes
: :	

_output_shapes
: 
?*
?
#__inference__traced_restore_5818206
file_prefix$
 assignvariableop_dense1_1_kernel$
 assignvariableop_1_dense1_1_bias&
"assignvariableop_2_dense2_1_kernel$
 assignvariableop_3_dense2_1_bias&
"assignvariableop_4_dense3_1_kernel$
 assignvariableop_5_dense3_1_bias%
!assignvariableop_6_probs_1_kernel#
assignvariableop_7_probs_1_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense1_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense1_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense2_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense2_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense3_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense3_1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_probs_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_probs_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7?
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
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
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
: :

_output_shapes
: :

_output_shapes
: 
?

?
)__inference_model_3_layer_call_fn_5818039
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_58178882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
D__inference_model_3_layer_call_and_return_conditional_losses_5817888

inputs
inputs_1
dense1_5817867
dense1_5817869
dense2_5817872
dense2_5817874
dense3_5817877
dense3_5817879
probs_5817882
probs_5817884
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense3/StatefulPartitionedCall?probs/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_5817867dense1_5817869*
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
C__inference_dense1_layer_call_and_return_conditional_losses_58176902 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_5817872dense2_5817874*
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
GPU 2J 8*L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_58177172 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_5817877dense3_5817879*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_58177432 
dense3/StatefulPartitionedCall?
probs/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0probs_5817882probs_5817884*
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
GPU 2J 8*K
fFRD
B__inference_probs_layer_call_and_return_conditional_losses_58177702
probs/StatefulPartitionedCall?
IdentityIdentity&probs/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2>
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
: :

_output_shapes
: :	

_output_shapes
: 
?

?
%__inference_signature_wrapper_5817931	
delta	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputdeltaunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_58176742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
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
: :

_output_shapes
: :	

_output_shapes
: 
?
?
C__inference_dense1_layer_call_and_return_conditional_losses_5818050

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
C__inference_dense2_layer_call_and_return_conditional_losses_5817717

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
?
?
C__inference_dense1_layer_call_and_return_conditional_losses_5817690

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
C__inference_dense2_layer_call_and_return_conditional_losses_5818070

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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?-
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
=__call__
>_default_save_signature
*?&call_and_return_all_conditional_losses"?*
_tf_keras_model?*{"class_name": "Model", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "dtype": "float32", "units": 144, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}, "name": "delta", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "probs", "inbound_nodes": [[["dense3", 0, 0, {}]]]}], "input_layers": [["input", 0, 0], ["delta", 0, 0]], "output_layers": [["probs", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "dtype": "float32", "units": 144, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}, "name": "delta", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "probs", "inbound_nodes": [[["dense3", 0, 0, {}]]]}], "input_layers": [["input", 0, 0], ["delta", 0, 0]], "output_layers": [["probs", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
B__call__
*C&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 72, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense3", "trainable": true, "dtype": "float32", "units": 144, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "delta", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "delta"}}
?

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "probs", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 144]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$layer_metrics
trainable_variables
	variables
%metrics
&non_trainable_variables

'layers
	regularization_losses
(layer_regularization_losses
=__call__
>_default_save_signature
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
!:$2dense1_1/kernel
:$2dense1_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)layer_metrics
trainable_variables
	variables
*metrics
+non_trainable_variables

,layers
regularization_losses
-layer_regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
!:$H2dense2_1/kernel
:H2dense2_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.layer_metrics
trainable_variables
	variables
/metrics
0non_trainable_variables

1layers
regularization_losses
2layer_regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
": 	H?2dense3_1/kernel
:?2dense3_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3layer_metrics
trainable_variables
	variables
4metrics
5non_trainable_variables

6layers
regularization_losses
7layer_regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
!:	?2probs_1/kernel
:2probs_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8layer_metrics
 trainable_variables
!	variables
9metrics
:non_trainable_variables

;layers
"regularization_losses
<layer_regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
?2?
)__inference_model_3_layer_call_fn_5817860
)__inference_model_3_layer_call_fn_5818017
)__inference_model_3_layer_call_fn_5817907
)__inference_model_3_layer_call_fn_5818039?
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
"__inference__wrapped_model_5817674?
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
D__inference_model_3_layer_call_and_return_conditional_losses_5817963
D__inference_model_3_layer_call_and_return_conditional_losses_5817995
D__inference_model_3_layer_call_and_return_conditional_losses_5817812
D__inference_model_3_layer_call_and_return_conditional_losses_5817787?
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
(__inference_dense1_layer_call_fn_5818059?
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
C__inference_dense1_layer_call_and_return_conditional_losses_5818050?
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
(__inference_dense2_layer_call_fn_5818079?
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
C__inference_dense2_layer_call_and_return_conditional_losses_5818070?
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
(__inference_dense3_layer_call_fn_5818098?
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
C__inference_dense3_layer_call_and_return_conditional_losses_5818089?
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
'__inference_probs_layer_call_fn_5818118?
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
B__inference_probs_layer_call_and_return_conditional_losses_5818109?
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
7B5
%__inference_signature_wrapper_5817931deltainput?
"__inference__wrapped_model_5817674?T?Q
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
C__inference_dense1_layer_call_and_return_conditional_losses_5818050\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????$
? {
(__inference_dense1_layer_call_fn_5818059O/?,
%?"
 ?
inputs?????????
? "??????????$?
C__inference_dense2_layer_call_and_return_conditional_losses_5818070\/?,
%?"
 ?
inputs?????????$
? "%?"
?
0?????????H
? {
(__inference_dense2_layer_call_fn_5818079O/?,
%?"
 ?
inputs?????????$
? "??????????H?
C__inference_dense3_layer_call_and_return_conditional_losses_5818089]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0??????????
? |
(__inference_dense3_layer_call_fn_5818098P/?,
%?"
 ?
inputs?????????H
? "????????????
D__inference_model_3_layer_call_and_return_conditional_losses_5817787?\?Y
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
D__inference_model_3_layer_call_and_return_conditional_losses_5817812?\?Y
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
D__inference_model_3_layer_call_and_return_conditional_losses_5817963?b?_
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
D__inference_model_3_layer_call_and_return_conditional_losses_5817995?b?_
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
)__inference_model_3_layer_call_fn_5817860?\?Y
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
)__inference_model_3_layer_call_fn_5817907?\?Y
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
)__inference_model_3_layer_call_fn_5818017?b?_
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
)__inference_model_3_layer_call_fn_5818039?b?_
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
B__inference_probs_layer_call_and_return_conditional_losses_5818109]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_probs_layer_call_fn_5818118P0?-
&?#
!?
inputs??????????
? "???????????
%__inference_signature_wrapper_5817931?a?^
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