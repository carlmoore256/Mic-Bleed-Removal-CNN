# ===== RUN EXPERIMENT ===================

# experiment = Experiment(api_key="OKhPlin1BVQJFzniHu1f3K1t3",
#                         project_name="micplacementwavenet", workspace="cm5409a")

# Hyperparameters --------
batch_size = 32
num_epochs = 20
Fc = 24 # num filters per layer (which is multiplied by depth)
sources_to_estimate = 1

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

downsample_0 = Conv1D(filters=Fc*1, kernel_size=15, padding='same')(input_layer)
downsample_0 = LeakyReLU(alpha=0.05)(downsample_0) # ACTIVATION
downsample_0 = MaxPooling1D(pool_size=2)(downsample_0) # DOWNSAMPLE

downsample_1 = Conv1D(filters=Fc*2, kernel_size=15, padding='same')(downsample_0)
downsample_1 = LeakyReLU(alpha=0.05)(downsample_1) # ACTIVATION
downsample_1 = MaxPooling1D(pool_size=2)(downsample_1) # DOWNSAMPLE

downsample_2 = Conv1D(filters=Fc*3, kernel_size=15, padding='same')(downsample_1)
downsample_2 = LeakyReLU(alpha=0.05) (downsample_2)# ACTIVATION
downsample_2 = MaxPooling1D(pool_size=2)(downsample_2) # DOWNSAMPLE

downsample_3 = Conv1D(filters=Fc*4, kernel_size=15, padding='same')(downsample_2)
downsample_3 = LeakyReLU(alpha=0.05) (downsample_3)# ACTIVATION
downsample_3 = MaxPooling1D(pool_size=2)(downsample_3) # DOWNSAMPLE

downsample_4 = Conv1D(filters=Fc*5, kernel_size=15, padding='same')(downsample_3)
downsample_4 = LeakyReLU(alpha=0.05) (downsample_4)# ACTIVATION
downsample_4 = MaxPooling1D(pool_size=2)(downsample_4) # DOWNSAMPLE

downsample_5 = Conv1D(filters=Fc*6, kernel_size=15, padding='same')(downsample_4)
downsample_5 = LeakyReLU(alpha=0.05) (downsample_5)# ACTIVATION
downsample_5 = MaxPooling1D(pool_size=2)(downsample_5) # DOWNSAMPLE

downsample_6 = Conv1D(filters=Fc*7, kernel_size=15, padding='same')(downsample_5)
downsample_6 = LeakyReLU(alpha=0.05) (downsample_6)# ACTIVATION
downsample_6 = MaxPooling1D(pool_size=2)(downsample_6) # DOWNSAMPLE

downsample_7 = Conv1D(filters=Fc*8, kernel_size=15, padding='same')(downsample_6)
downsample_7 = LeakyReLU(alpha=0.05) (downsample_7)# ACTIVATION
downsample_7 = MaxPooling1D(pool_size=2)(downsample_7) # DOWNSAMPLE

downsample_8 = Conv1D(filters=Fc*9, kernel_size=15, padding='same')(downsample_7)
downsample_8 = LeakyReLU(alpha=0.05) (downsample_8)# ACTIVATION
downsample_8 = MaxPooling1D(pool_size=2)(downsample_8) # DOWNSAMPLE

downsample_9 = Conv1D(filters=Fc*10, kernel_size=15, padding='same')(downsample_8)
downsample_9 = LeakyReLU(alpha=0.05) (downsample_9)# ACTIVATION
downsample_9 = MaxPooling1D(pool_size=2)(downsample_9) # DOWNSAMPLE

downsample_10 = Conv1D(filters=Fc*11, kernel_size=15, padding='same')(downsample_9)
downsample_10 = LeakyReLU(alpha=0.05) (downsample_10)# ACTIVATION
downsample_10 = MaxPooling1D(pool_size=2)(downsample_10) # DOWNSAMPLE

downsample_11 = Conv1D(filters=256, kernel_size=15, padding='same')(downsample_10)
downsample_11 = LeakyReLU(alpha=0.05) (downsample_11)# ACTIVATION
downsample_11 = MaxPooling1D(pool_size=2)(downsample_11) # DOWNSAMPLE


# =====================================================
# consider extending this so that shape in center reaches 4 or even 2 (12 layer)

resized = Reshape((32,32,1))(downsample_11)
center = Conv2D(filters=Fc*6, kernel_size=5, padding='same')(resized)

upsample_0 = Conv2D(filters=Fc*7, kernel_size=5, padding='same', activation='relu')(center)
# upsample_0 = LeakyReLU(alpha=0.3)(upsample_0)
upsample_0 = UpSampling2D(size=(2,2))(upsample_0)

upsample_1 = Conv2D(filters=Fc*5, kernel_size=5, padding='same', activation='relu')(upsample_0)
# upsample_1 = LeakyReLU(alpha=0.3)(upsample_1)

upsample_2 = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(upsample_1)
# upsample_2 = LeakyReLU(alpha=0.3)(upsample_2)
print(upsample_2.get_shape)
# 1 to 1024
# output_reshape = Reshape((512, 256, 1))(upsample_2)
output_reshape = Reshape((8, 16384, 1))(upsample_2)

# output_layer = concatenate([input_layer, upsample_0])
output_layer = Conv2D(filters=sources_to_estimate, kernel_size=3, padding='same', activation='relu')(output_reshape)

model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

# cb = callback(x_val, y_val, model, num_tests=1)

result = model.fit(x_train,
                   y_train,
                   batch_size=batch_size,
                   shuffle=True,
                   epochs=num_epochs)
                  #  validation_data=(x_val, y_val))
                  #  callbacks=[cb])
