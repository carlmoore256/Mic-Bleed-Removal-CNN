# ===== RUN EXPERIMENT ===================

experiment = Experiment(api_key="OKhPlin1BVQJFzniHu1f3K1t3",
                        project_name="micplacementwavenet", workspace="cm5409a")

# Hyperparameters --------
batch_size = 32
num_epochs = 5
Fc = 24 # num filters per layer (which is multiplied by depth)
sources_to_estimate = 1
LeakyReluAlpha = 0.2
padding = 'same' # try 'causal'
dilation_rate = 2
pool = False

# -----------------------

init_x = x_train[0]
init_y = y_train[0]

input_shape = (x_train.shape[1], 1)
output_shape = (y_train.shape[1], 1)
main_dim = input_shape[1]

# MODEL =======================================================
# wave-U-net keras implementation, details: https://arxiv.org/pdf/1806.03185.pdf

# NOTES:
# change padding from 'same' to 'causal'
input_layer = Input(shape=input_shape)

downsample_0 = Conv1D(filters=Fc*1, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(input_layer)
downsample_0 = LeakyReLU(alpha=LeakyReluAlpha)(downsample_0) # ACTIVATION
# downsample_0 = MaxPooling1D(pool_size=2)(downsample_0) # DOWNSAMPLE
if pool:
  downsample_0 = AveragePooling1D(pool_size=2)(downsample_0) # DOWNSAMPLE

downsample_1 = Conv1D(filters=Fc*2, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_0)
downsample_1 = LeakyReLU(alpha=LeakyReluAlpha)(downsample_1) # ACTIVATION
# downsample_1 = MaxPooling1D(pool_size=2)(downsample_1) # DOWNSAMPLE
if pool:
  downsample_1 = AveragePooling1D(pool_size=2)(downsample_1) # DOWNSAMPLE

downsample_2 = Conv1D(filters=Fc*3, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_1)
downsample_2 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_2)# ACTIVATION
# downsample_2 = MaxPooling1D(pool_size=2)(downsample_2) # DOWNSAMPLE
if pool:
  downsample_2 = AveragePooling1D(pool_size=2)(downsample_2) # DOWNSAMPLE

downsample_3 = Conv1D(filters=Fc*4, kernel_size=15, padding=padding)(downsample_2)
downsample_3 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_3)# ACTIVATION
# downsample_3 = MaxPooling1D(pool_size=2)(downsample_3) # DOWNSAMPLE
if pool:
  downsample_3 = AveragePooling1D(pool_size=2)(downsample_3) # DOWNSAMPLE

downsample_4 = Conv1D(filters=Fc*5, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_3)
downsample_4 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_4)# ACTIVATION
# downsample_4 = MaxPooling1D(pool_size=2)(downsample_4) # DOWNSAMPLE
if pool:
  downsample_4 = AveragePooling1D(pool_size=2)(downsample_4) # DOWNSAMPLE

downsample_5 = Conv1D(filters=Fc*6, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_4)
downsample_5 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_5)# ACTIVATION
# downsample_5 = MaxPooling1D(pool_size=2)(downsample_5) # DOWNSAMPLE
if pool:
  downsample_5 = AveragePooling1D(pool_size=2)(downsample_5) # DOWNSAMPLE

downsample_6 = Conv1D(filters=Fc*7, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_5)
downsample_6 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_6)# ACTIVATION
# downsample_6 = MaxPooling1D(pool_size=2)(downsample_6) # DOWNSAMPLE
if pool:
  downsample_6 = AveragePooling1D(pool_size=2)(downsample_6) # DOWNSAMPLE

downsample_7 = Conv1D(filters=Fc*8, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_6)
downsample_7 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_7)# ACTIVATION
# downsample_7 = MaxPooling1D(pool_size=2)(downsample_7) # DOWNSAMPLE
if pool:
  downsample_7 = AveragePooling1D(pool_size=2)(downsample_7) # DOWNSAMPLE

downsample_8 = Conv1D(filters=Fc*9, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_7)
downsample_8 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_8)# ACTIVATION
# downsample_8 = MaxPooling1D(pool_size=2)(downsample_8) # DOWNSAMPLE
if pool:
  downsample_8 = AveragePooling1D(pool_size=2)(downsample_8) # DOWNSAMPLE

downsample_9 = Conv1D(filters=Fc*10, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_8)
downsample_9 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_9)# ACTIVATION
# downsample_9 = MaxPooling1D(pool_size=2)(downsample_9) # DOWNSAMPLE
if pool:
  downsample_9 = AveragePooling1D(pool_size=2)(downsample_9) # DOWNSAMPLE

downsample_10 = Conv1D(filters=Fc*11, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_9)
downsample_10 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_10)# ACTIVATION
# downsample_10 = MaxPooling1D(pool_size=2)(downsample_10) # DOWNSAMPLE
if pool:
  downsample_10 = AveragePooling1D(pool_size=2)(downsample_10) # DOWNSAMPLE

downsample_11 = Conv1D(filters=Fc*12, kernel_size=15, padding=padding, dilation_rate=dilation_rate)(downsample_10)
downsample_11 = LeakyReLU(alpha=LeakyReluAlpha) (downsample_11)# ACTIVATION
# downsample_11 = MaxPooling1D(pool_size=2)(downsample_11) # DOWNSAMPLE
if pool:
  downsample_11 = AveragePooling1D(pool_size=2)(downsample_11) # DOWNSAMPLE


# =====================================================
# consider extending this so that shape in center reaches 4 or even 2 (12 layer)

upsample_11 = Conv1D(filters=Fc*12, kernel_size=5, padding=padding)(downsample_11)
upsample_11 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_11) # ACTIVATION
upsample_11 = UpSampling1D(size=2)(upsample_11) # UPSAMPLE

upsample_10 = concatenate([upsample_11, downsample_10])
upsample_10 = Conv1D(filters=Fc*11, kernel_size=5, padding=padding)(upsample_10)
upsample_10 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_10) # ACTIVATION
upsample_10 = UpSampling1D(size=2)(upsample_10) # UPSAMPLE

upsample_9 = concatenate([upsample_10, downsample_9])
upsample_9 = Conv1D(filters=Fc*10, kernel_size=5, padding=padding)(upsample_9)
upsample_9 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_9) # ACTIVATION
upsample_9 = UpSampling1D(size=2)(upsample_9) # UPSAMPLE

upsample_8 = concatenate([upsample_9, downsample_8])
upsample_8 = Conv1D(filters=Fc*9, kernel_size=5, padding=padding)(upsample_8)
upsample_8 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_8) # ACTIVATION
upsample_8 = UpSampling1D(size=2)(upsample_8) # UPSAMPLE

upsample_7 = concatenate([upsample_8, downsample_7])
upsample_7 = Conv1D(filters=Fc*8, kernel_size=5, padding=padding)(upsample_7)
upsample_7 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_7) # ACTIVATION
upsample_7 = UpSampling1D(size=2)(upsample_7) # UPSAMPLE

upsample_6 = concatenate([upsample_7, downsample_6])
upsample_6 = Conv1D(filters=Fc*7, kernel_size=5, padding=padding)(upsample_6)
upsample_6 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_6) # ACTIVATION
upsample_6 = UpSampling1D(size=2)(upsample_6) # UPSAMPLE

upsample_5 = concatenate([upsample_6, downsample_5])
upsample_5 = Conv1D(filters=Fc*6, kernel_size=5, padding=padding)(upsample_5)
upsample_5 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_5) # ACTIVATION
upsample_5 = UpSampling1D(size=2)(upsample_5) # UPSAMPLE

upsample_4 = concatenate([upsample_5, downsample_4])
upsample_4 = Conv1D(filters=Fc*5, kernel_size=5, padding=padding)(upsample_4)
upsample_4 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_4) # ACTIVATION
upsample_4 = UpSampling1D(size=2)(upsample_4) # UPSAMPLE

upsample_3 = concatenate([upsample_4, downsample_3])
upsample_3 = Conv1D(filters=Fc*4, kernel_size=5, padding=padding)(upsample_3)
upsample_3 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_3) # ACTIVATION
upsample_3 = UpSampling1D(size=2)(upsample_3) # UPSAMPLE

upsample_2 = concatenate([upsample_3, downsample_2])
upsample_2 = Conv1D(filters=Fc*3, kernel_size=5, padding=padding)(upsample_2)
upsample_2 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_2) # ACTIVATION
upsample_2 = UpSampling1D(size=2)(upsample_2) # UPSAMPLE

upsample_1 = concatenate([upsample_2, downsample_1])
upsample_1 = Conv1D(filters=Fc*2, kernel_size=5, padding=padding)(upsample_1)
upsample_1 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_1) # ACTIVATION
upsample_1 = UpSampling1D(size=2)(upsample_1) # UPSAMPLE

upsample_0 = concatenate([upsample_1, downsample_0]) # CONCATENATE SKIP
upsample_0 = Conv1D(filters=Fc*1, kernel_size=1, padding=padding)(upsample_0)
upsample_0 = LeakyReLU(alpha=LeakyReluAlpha)(upsample_0) # ACTIVATION
upsample_0 = UpSampling1D(size=2)(upsample_0) # UPSAMPLE

output_layer = concatenate([input_layer, upsample_0])
output_layer = Conv1D(filters=sources_to_estimate, kernel_size=1, padding=padding, activation='tanh')(output_layer)

model = Model(input_layer, output_layer)
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['mean_squared_error', 'logcosh', 'accuracy'])
model.summary()

# keras.utils.plot_model(model, to_file='model_2.png', show_shapes=True)

cb = callback(x_val, y_val, model, num_tests=1)

result = model.fit(x_train,
                   y_train,
                   batch_size=batch_size,
                   shuffle=True,
                   epochs=num_epochs,
                   validation_data=(x_val, y_val),
                   callbacks=[cb])
