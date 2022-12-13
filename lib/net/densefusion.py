from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, Concatenate


class DenseFusionNetParams:
    def __init__(self):
        self.num_embeddings = 128
        self.conv1d_kernel = 1
        self.rgb_conv1d_dim = 256
        self.pcl_conv1d_dim = 256
        self.rgbd_feats1_conv1d_dim = 512
        self.rgbd_feats2_conv1d_dim = 1024


class DenseFusionNet:
    def __init__(self, params: DenseFusionNetParams):
        self.params = params

    def densefusion_layers(self, rgb_emb, pcl_emb):

        rgb_conv = Conv1D(filters=self.params.rgb_conv1d_dim,
                          kernel_size=self.params.conv1d_kernel, activation='relu')(rgb_emb)
        pcl_conv = Conv1D(filters=self.params.pcl_conv1d_dim,
                          kernel_size=self.params.conv1d_kernel, activation='relu')(pcl_emb)

        features_1 = Concatenate()([rgb_emb, pcl_emb])
        features_2 = Concatenate()([rgb_conv, pcl_conv])

        rgbd_features_1 = Conv1D(filters=self.params.rgbd_feats1_conv1d_dim,
                                 kernel_size=self.params.conv1d_kernel, activation='relu')(features_1)
        rgbd_features_2 = Conv1D(filters=self.params.rgbd_feats2_conv1d_dim,
                                 kernel_size=self.params.conv1d_kernel, activation='relu')(rgbd_features_1)

        final_features = Concatenate()([features_1, features_2, rgbd_features_2])
        return final_features

    def build_dense_fusion_model(self, rgb_emb_shape, pcl_emb_shape):
        rgb_emb_input = Input(shape=rgb_emb_shape, name='rgb_emb_input')
        pcl_emb_input = Input(shape=pcl_emb_shape, name='pcl_emb_input')
        final_features = self.densefusion_layers(rgb_emb_input, pcl_emb_input)
        model = Model(inputs=[rgb_emb_input, pcl_emb_input], outputs=final_features, name='dense_fusion_model')
        return model

    def dense_connection_layers(self, pcl_emb):

        pcl_conv = Conv1D(filters=self.params.pcl_conv1d_dim,
                          kernel_size=self.params.conv1d_kernel, activation='relu')(pcl_emb)

        pcld_features_1 = Conv1D(filters=self.params.rgbd_feats1_conv1d_dim,
                                 kernel_size=self.params.conv1d_kernel, activation='relu')(pcl_emb)
        pcld_features_2 = Conv1D(filters=self.params.rgbd_feats2_conv1d_dim,
                                 kernel_size=self.params.conv1d_kernel, activation='relu')(pcld_features_1)

        final_features = Concatenate()([pcl_emb, pcl_conv, pcld_features_2])
        return final_features

    def build_dense_model(self, pcl_emb_shape):
        pcl_emb_input = Input(shape=pcl_emb_shape, name='pcl_emb_input')
        final_features = self.dense_connection_layers(pcl_emb_input)
        model = Model(inputs=pcl_emb_input, outputs=final_features, name='dense_model')
        return model
