# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_segmentation.html

library(torch)
library(torchvision)
library(luz)
library(torchdatasets)
library(raster)

# define the model --------------------------------------------------------

set.seed(34346)


# MobileNetV2 paper: https://arxiv.org/abs/1801.04381
encoder <- nn_module(
  initialize = function() {
    model <- model_mobilenet_v2(pretrained = TRUE)
    self$stages <- nn_module_list(list(
      nn_identity(),
      model$features[1:2],
      model$features[3:4],
      model$features[5:7],
      model$features[8:14],
      model$features[15:18]
    ))
    for (par in self$parameters) {
      par$requires_grad_(FALSE)
    }
  },
  forward = function(x) {
    features <- list()
    for (i in 1:length(self$stages)) {
      x <- self$stages[[i]](x)
      features[[length(features) + 1]] <- x
    }
    features
  }
)

decoder_block <- nn_module(
  initialize = function(in_channels,
                        skip_channels,
                        out_channels) {
    self$upsample <- nn_conv_transpose2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 2,
      stride = 2
    )
    self$activation <- nn_relu()
    self$conv <- nn_conv2d(
      in_channels = out_channels + skip_channels,
      out_channels = out_channels,
      kernel_size = 3,
      padding = "same"
    )
  },
  forward = function(x, skip) {
    x <- x |>
      self$upsample() |>
      self$activation()
    input <- torch_cat(list(x, skip), dim = 2)
    input |>
      self$conv() |>
      self$activation()
  }
)

decoder <- nn_module(
  initialize = function(
    decoder_channels = c(256, 128, 64, 32, 16),
    encoder_channels = c(16, 24, 32, 96, 320)) {
    encoder_channels <- rev(encoder_channels)
    skip_channels <- c(encoder_channels[-1], 3)
    in_channels <- c(encoder_channels[1], decoder_channels)

    depth <- length(encoder_channels)

    self$blocks <- nn_module_list()
    for (i in seq_len(depth)) {
      self$blocks$append(decoder_block(
        in_channels = in_channels[i],
        skip_channels = skip_channels[i],
        out_channels = decoder_channels[i]
      ))
    }
  },
  forward = function(features) {
    features <- rev(features)
    x <- features[[1]]
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x, features[[i + 1]])
    }
    x
  }
)

model <- nn_module(
  initialize = function() {
    self$encoder <- encoder()
    self$decoder <- decoder()
    self$output <- nn_conv2d(
      in_channels = 16,
      out_channels = 3,
      kernel_size = 3,
      padding = "same"
    )
  },
  forward = function(x) {
    x |>
      self$encoder() |>
      self$decoder() |>
      self$output()
  }
)


# set up the data ---------------------------------------------------------

set.seed(89743)

data_dir <- here::here("torch-datasets")
do_data_download <- !dir.exists(data_dir)
ds <- oxford_pet_dataset(root = data_dir, download = do_data_download)

initialise_pet <- function(
    ...,
    size,
    normalize = TRUE,
    augmentation = NULL
) {

  self$augmentation <- augmentation
  input_transform <- function(x) {
    x <- x |>
      transform_to_tensor() |>
      transform_resize(size)
    if (normalize) {
      x <- x |>
        transform_normalize(
          mean = c(0.485, 0.456, 0.406),
          std = c(0.229, 0.224, 0.225)
        )
    }
    x
  }

  target_transform <- function(x) {
    x <- torch_tensor(x, dtype = torch_long())
    x <- x[newaxis, ..]
    # interpolation = 0 makes sure we
    # still end up with integer classes
    x <- transform_resize(x, size, interpolation = 0)
    x[1, ..]
  }

  super$initialize(
    ...,
    transform = input_transform,
    target_transform = target_transform
  )
}

pet_dataset <- torch::dataset(
  inherit = oxford_pet_dataset,
  initialize = initialise_pet,
  .getitem = function(i) {
    item <- super$.getitem(i)
    if (!is.null(self$augmentation)) {
      self$augmentation(item)
    } else {
      list(x = item$x, y = item$y)
    }
  }
)

augmentation <- function(item) {
  vflip <- runif(1) > 0.5
  x <- item$x
  y <- item$y
  if (vflip) {
    x <- transform_vflip(x)
    y <- transform_vflip(y)
  }
  list(x = x, y = y)
}


# set up training and test ------------------------------------------------

set.seed(89723)

train_ds <- pet_dataset(
  root = data_dir,
  split = "train",
  size = c(224, 224),
  augmentation = augmentation
)

valid_ds <- pet_dataset(
  root = data_dir,
  split = "valid",
  size = c(224, 224)
)

train_dl <- dataloader(
  train_ds,
  batch_size = 32,
  shuffle = TRUE
)

valid_dl <- dataloader(
  valid_ds,
  batch_size = 32
)



# training ----------------------------------------------------------------

set.seed(13453)

model <- model |>
  setup(
    optimizer = optim_adam,
    loss = nn_cross_entropy_loss()
  )

do_search_learning_rate <- FALSE
if(do_search_learning_rate) {
  rates_and_losses <- model |> lr_finder(train_dl)
  rates_and_losses |> plot()
}

fitted_model_path <- here::here("image-seg", "pet-segmentation-fitted.rds")
do_model_training <- !file.exists(fitted_model_path)
if(do_model_training) {
  fitted <- model |>
    fit(
      train_dl,
      epochs = 20,
      valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 2),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = 20,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end")
      ),
      verbose = TRUE
    )
  luz_save(fitted, path = fitted_model_path)

} else {
  fitted <- luz_load(fitted_model_path)
}


# plot the results --------------------------------------------------------

set.seed(34452)

valid_ds_4plot <- pet_dataset(
  root = data_dir,
  split = "valid",
  size = c(224, 224),
  normalize = FALSE
)

indices <- 1:16

preds <- predict(
  fitted,
  dataloader(dataset_subset(valid_ds, indices))
)

png(
  here::here("image-seg", "pet-segmentation.png"),
  width = 1200,
  height = 1200,
  bg = "black"
)

par(mfcol = c(4, 4), mar = rep(1, 4))

for (i in indices) {
  mask <- as.array(
    torch_argmax(preds[i, ..], 1)$to(device = "cpu")
  )
  mask <- raster::ratify(raster::raster(mask))

  img <- as.array(valid_ds_4plot[i][[1]]$permute(c(2, 3, 1)))
  cond <- img > 0.99999
  img[cond] <- 0.99999
  img <- raster::brick(img)

  # plot image
  raster::plotRGB(img, scale = 1, asp = 1, margins = TRUE)

  # overlay mask
  plot(
    mask,
    alpha = 0.4,
    legend = FALSE,
    axes = FALSE,
    add = TRUE
  )
}

dev.off()

