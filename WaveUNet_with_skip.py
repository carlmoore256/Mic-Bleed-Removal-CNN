# experiment = Experiment(api_key="OKhPlin1BVQJFzniHu1f3K1t3",
#                         project_name="micplacementwavenet", workspace="cm5409a")

batch_size = 32
steps_per_epoch = 1000
num_epochs = 20

# gen_obj_train = TrainGenerator(x_train, y_train, win_size, batch_size)
# gen_obj_val = TrainGenerator(x_val, y_val, win_size, batch_size)


# train_gen = gen_obj_train.generator()
# val_gen = gen_obj_val.generator()

# init_x, init_y = next(train_gen)

Fc = 24 #num filters per layer * depth

init_x = x_train[0]
init_y = y_train[0]

input_shape = (x_train.shape[1], 1)
output_shape = (y_train.shape[1], 1)
main_dim = input_shape[1]

#INPUT =======================================================
input_layer = Input(shape=input_shape)

downsample_0 = Conv1D(filters=Fc*1, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(input_layer)
downsample_0 = LeakyReLU(alpha=0.05)(downsample_0) # ACTIVATION
downsample_0 = MaxPooling1D(pool_size=2)(downsample_0) # DOWNSAMPLE

downsample_1 = Conv1D(filters=Fc*2, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_0)
downsample_1 = LeakyReLU(alpha=0.05)(downsample_1) # ACTIVATION
downsample_1 = MaxPooling1D(pool_size=2)(downsample_1) # DOWNSAMPLE

downsample_2 = Conv1D(filters=Fc*3, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_1)
downsample_2 = LeakyReLU(alpha=0.05) (downsample_2)# ACTIVATION
downsample_2 = MaxPooling1D(pool_size=2)(downsample_2) # DOWNSAMPLE

downsample_3 = Conv1D(filters=Fc*4, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_2)
downsample_3 = LeakyReLU(alpha=0.05) (downsample_3)# ACTIVATION
downsample_3 = MaxPooling1D(pool_size=2)(downsample_3) # DOWNSAMPLE

downsample_4 = Conv1D(filters=Fc*5, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_3)
downsample_4 = LeakyReLU(alpha=0.05) (downsample_4)# ACTIVATION
downsample_4 = MaxPooling1D(pool_size=2)(downsample_4) # DOWNSAMPLE

downsample_5 = Conv1D(filters=Fc*6, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_4)
downsample_5 = LeakyReLU(alpha=0.05) (downsample_5)# ACTIVATION
downsample_5 = MaxPooling1D(pool_size=2)(downsample_5) # DOWNSAMPLE

downsample_6 = Conv1D(filters=Fc*7, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_5)
downsample_6 = LeakyReLU(alpha=0.05) (downsample_6)# ACTIVATION
downsample_6 = MaxPooling1D(pool_size=2)(downsample_6) # DOWNSAMPLE

downsample_7 = Conv1D(filters=Fc*8, kernel_size=15, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_6)
downsample_7 = LeakyReLU(alpha=0.05) (downsample_7)# ACTIVATION
downsample_7 = MaxPooling1D(pool_size=2)(downsample_7) # DOWNSAMPLE

# =====================================================

upsample_7 = Conv1D(filters=Fc*8, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(downsample_7)
upsample_7 = LeakyReLU(alpha=0.05)(upsample_7) # ACTIVATION
upsample_7 = UpSampling1D(size=2)(upsample_7) # UPSAMPLE

upsample_5 = concatenate([upsample_7, downsample_6])
upsample_6 = Conv1D(filters=Fc*7, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_7)
upsample_6 = LeakyReLU(alpha=0.05)(upsample_6) # ACTIVATION
upsample_6 = UpSampling1D(size=2)(upsample_6) # UPSAMPLE


upsample_5 = concatenate([upsample_6, downsample_5])
upsample_5 = Conv1D(filters=Fc*6, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_5)
upsample_5 = LeakyReLU(alpha=0.05)(upsample_5) # ACTIVATION
upsample_5 = UpSampling1D(size=2)(upsample_5) # UPSAMPLE


upsample_4 = concatenate([upsample_5, downsample_4])
upsample_4 = Conv1D(filters=Fc*5, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_4)
upsample_4 = LeakyReLU(alpha=0.05)(upsample_4) # ACTIVATION
upsample_4 = UpSampling1D(size=2)(upsample_4) # UPSAMPLE


upsample_3 = concatenate([upsample_4, downsample_3])
upsample_3 = Conv1D(filters=Fc*4, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_3)
upsample_3 = LeakyReLU(alpha=0.05)(upsample_3) # ACTIVATION
upsample_3 = UpSampling1D(size=2)(upsample_3) # UPSAMPLE


upsample_2 = concatenate([upsample_3, downsample_2])
upsample_2 = Conv1D(filters=Fc*3, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_2)
upsample_2 = LeakyReLU(alpha=0.05)(upsample_2) # ACTIVATION
upsample_2 = UpSampling1D(size=2)(upsample_2) # UPSAMPLE


upsample_1 = concatenate([upsample_2, downsample_1])
upsample_1 = Conv1D(filters=Fc*2, kernel_size=5, padding='same',
                 input_shape=(1, init_x.shape[1]))(upsample_1)
upsample_1 = LeakyReLU(alpha=0.05)(upsample_1) # ACTIVATION
upsample_1 = UpSampling1D(size=2)(upsample_1) # UPSAMPLE


upsample_0 = concatenate([upsample_1, downsample_0]) # CONCATENATE SKIP
upsample_0 = Conv1D(filters=Fc*1, kernel_size=1, padding='same',
                         input_shape=(init_y.shape[1], 1))(upsample_0)
upsample_0 = LeakyReLU(alpha=0.05)(upsample_0) # ACTIVATION
upsample_0 = UpSampling1D(size=2)(upsample_0) # UPSAMPLE



# combined = concatenate([mag.output, phase.output])

upsample_0 = Dense(output_shape[1], activation='tanh')(upsample_0)

model = Model(input_layer, upsample_0)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


# cb = callback(gen_obj_val, model)

# result = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                            #  epochs=num_epochs, validation_data=val_gen, validation_steps=100, verbose=1,
                            #  callbacks=[cb])

result = model.fit(x_train,
                   y_train,
                   batch_size=batch_size,
                   shuffle=True,
                   epochs=num_epochs,
                   validation_data=(x_val, y_val))
