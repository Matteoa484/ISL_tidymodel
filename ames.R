# Libraries ---------------------------------------------------------------

# data library
library(AmesHousing)
# plots & tables
library(ggplot2)
library(patchwork)
library(kableExtra)
# data wrangling
library(dplyr)
# modeling
library(tidymodels)

tidymodels_prefer()



# Load data ---------------------------------------------------------------

# create cleaned ames data frame
ames <- make_ames()

# log-transform target variable
ames <- ames %>%
    mutate(Sale_Price = log10(Sale_Price))



# Train/test sets ---------------------------------------------------------

## training/testing sets
# set seed
set.seed(123)
# data set split
ames_split <- initial_split(ames, prop = .8, strata = Sale_Price)

# train
ames_train <- training(ames_split)
# test
ames_test <- testing(ames_split)

## cross validation
set.seed(234)
# cross-validation data set
ames_fold <- vfold_cv(ames_train, v = 10, strata = Sale_Price)



# Metrics & Misc ----------------------------------------------------------

# set which metrics to use
ames_met <- metric_set(rmse, rsq, mae)

# parallel computing
doParallel::registerDoParallel()



# Linear Regression -------------------------------------------------------

## Model spec
lr_spec <- linear_reg() %>%
    set_engine('lm') %>%
    set_mode('regression')

## Recipe
lr_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(Garage_Qual, Garage_Cond, Garage_Type, threshold = 0.05) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())
    
## workflows
lr_wf <- workflow() %>%
    add_model(lr_spec) %>%
    add_recipe(lr_rec)
  
## fits model
lr_fit <- fit(lr_wf, data = ames_train)

## get performance stats
preds <- augment(lr_fit, new_data = ames_train)

preds %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Ridge Regression --------------------------------------------------------

## Model spec
ridge_spec <- 
    linear_reg(mixture = 0, penalty = tune()) %>%
    set_mode('regression') %>%
    set_engine('glmnet')

## Recipe
ridge_rec <-
    recipe(Sale_Price ~ ., data = ames_train) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(), threshold = 0.05) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_predictors())



## Workflows
ridge_wf <- workflow() %>%
    add_model(ridge_spec) %>%
    add_recipe(ridge_rec)

## Tune

# create the 'penalty' grid search for tuning
penalty_grid <- grid_regular(penalty(range = c(-5, 5)),
                             levels = 50)

# tune
set.seed(111)
ridge_tune <- tune_grid(ridge_wf,
                        resamples = ames_fold,
                        grid      = penalty_grid)

# select model with best metric and fit it
best_ridge <- select_best(ridge_tune, metric = 'rmse') %>%
    finalize_workflow(x = ridge_wf, parameters = .) %>%
    fit(data = ames_train)


## Compare performance

# build named list
models <- list('linear regression' = lr_fit,
               'ridge regression'  = best_ridge)

# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
    group_by(model) %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Lasso Regression --------------------------------------------------------

## Model spec
lasso_spec <- 
    linear_reg(mixture = 1, penalty = tune()) %>%
    set_mode('regression') %>%
    set_engine('glmnet')

## Recipe
# it uses the same recipe as ridge

## Workflows
lasso_wf <- workflow() %>%
    add_recipe(ridge_rec) %>%
    add_model(lasso_spec)

## Tune

# create the 'penalty' grid search for tuning
penalty_grid <- grid_regular(penalty(range = c(-5, 5)),
                             levels = 50)

# tune
set.seed(222)
lasso_tune <- tune_grid(lasso_wf,
                        resamples = ames_fold,
                        grid      = penalty_grid)

# select model with best metric and fit it
best_lasso <- select_best(lasso_tune, metric = 'rmse') %>%
    finalize_workflow(x = lasso_wf, parameters = .) %>%
    fit(data = ames_train)


## Compare performance

# build named list
models <- append(models, list('lasso regression' = best_lasso))

# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
    group_by(model) %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Elastic Net -------------------------------------------------------------

## Model spec
elnet_spec <- 
    linear_reg(mixture = tune(), penalty = tune()) %>%
    set_mode('regression') %>%
    set_engine('glmnet')

## Recipe
# it uses the same recipe as ridge

## Workflows
elnet_wf <- workflow() %>%
    add_recipe(ridge_rec) %>%
    add_model(elnet_spec)

## Tune

# create the 'penalty' grid search for tuning
elnet_grid <- grid_regular(mixture(range = c(0, 1)),
                           penalty(range = c(-5, 5)),
                           levels = c(10, 40))

# tune
set.seed(333)
elnet_tune <- tune_grid(elnet_wf,
                        resamples = ames_fold,
                        grid      = elnet_grid)

# select model with best metric and fit it
best_elnet <- select_best(elnet_tune, metric = 'rmse') %>%
    finalize_workflow(x = elnet_wf, parameters = .) %>%
    fit(data = ames_train)

## Compare performance

# build named list
models <- append(models, list('elastic net' = best_elnet))

# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
    group_by(model) %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Principal Component Regression ------------------------------------------

## Model spec

# use the same model spec as linear regression
pcr_spec <- lr_spec

## Recipe
pcr_rec <-
    recipe(Sale_Price ~ ., data = ames_train) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_predictors()) %>%
    step_pca(all_predictors(), threshold = tune())

## Workflows
pcr_wf <- workflow() %>%
    add_model(pcr_spec) %>%
    add_recipe(pcr_rec)

## Tune

# create the 'penalty' grid search for tuning
pcr_grid <- grid_regular(threshold(), levels = 10)

# tune
set.seed(444)
pcr_tune <- tune_grid(pcr_wf,
                      resamples = ames_fold,
                      grid      = pcr_grid)


# select model with best metric and fit it
best_pcr <- select_best(pcr_tune, metric = 'rmse') %>%
  finalize_workflow(x = pcr_wf, parameters = .) %>%
  fit(data = ames_train)

## Compare performance

# build named list
models <- append(models, list('Principal Component Reg' = best_pcr))

# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
    group_by(model) %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Partial Least Squared ---------------------------------------------------

## Model spec

# use the same model spec as linear regression
# renaming it for clearer code
pls_spec <- lr_spec


## Recipe
pls_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_predictors()) %>%
    step_pls(all_predictors(), num_comp = tune(), outcome = 'Sale_Price')


## workflow
pls_wf <- workflow() %>%
  add_model(pls_spec) %>%
  add_recipe(pls_rec)

pls_grid <- grid_regular(num_comp(c(1, 20)), levels = 10)

pls_tune <- tune_grid(pls_wf,
                      resamples = ames_fold,
                      grid      = pls_grid)



# Random Forest -----------------------------------------------------------

## Model spec
randf_spec <- rand_forest(mtry = 27) %>%
    set_engine('randomForest') %>%
    set_mode('regression')

## Recipe
randf_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_novel(all_nominal_predictors())

## Workflows
randf_wf <- workflow() %>%
    add_model(randf_spec) %>%
    add_recipe(randf_rec)

## Tune

# create the 'penalty' grid search for tuning


randf_fit <- fit(randf_wf, data = ames_train)



models <- append(models, list('random forest' = randf_fit))
