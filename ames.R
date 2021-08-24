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
ames_met <- metric_set(rmse, mae)

# parallel computing
doParallel::registerDoParallel()



# Test EDA plots & tables -------------------------------------------------

## Function to plot the frequency of nominal variables
nom_bar_plot <- function(dataset, col) {
  
    if(is.character(col)) {
        col <- sym(col)
    } else {
        col <- enquo(col)
    }
    
    ggplot(dataset, aes(y = forcats::fct_infreq(!!col))) +
        geom_bar(fill = 'steelblue', alpha = .5) +
        labs(title = glue::glue('{col}'), x = '', y = '') +
        theme_minimal()
  
}


targ_num_plot <- function(dataset, target_var, other_var) {
  
    t_var <- enquo(target_var)
    o_var <- enquo(other_var)
    
    ggplot(dataset, aes(x = !!o_var, y = !!t_var)) +
        geom_point(alpha = .5)
  
  
}


c('Garage_Type', 'Garage_Finish', 'Garage_Qual', 'Garage_Cond') %>%
    purrr::map(~nom_bar_plot(dataset = ames_train, col = .)) %>%
    patchwork::wrap_plots()


c('MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Lot_Shape') %>%
  purrr::map(~nom_bar_plot(dataset = ames_train, col = .)) %>%
  patchwork::wrap_plots()

nom_bar_plot(ames_train, 'Neighborhood')

targ_num_plot(ames_train, Sale_Price, Total_Bsmt_SF)


targ_num_plot(ames_train, Sale_Price, First_Flr_SF) + 
    scale_x_log10()

targ_num_plot(ames_train, Sale_Price, Pool_Area)



ames_train %>%
    ggplot(aes(x = Sale_Price, y = Gr_Liv_Area)) +
    geom_point(alpha = .5)

ames_train %>%
  ggplot(aes(x = Sale_Price)) +
  geom_histogram(alpha = .7) +
  facet_wrap(~ Overall_Qual) +
    scale_x_continuous(labels = scales::number_format())


ames_train %>%
    naniar::vis_miss()

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
    

## recipe prop
test_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Qual, Garage_Cond, Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.05) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


## recipe prop2
test2_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Qual, Garage_Cond, Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.02) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

## recipe prop3
test3_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_rm(Garage_Qual) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Cond, Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.02) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


## recipe prop4
test4_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_rm(Garage_Qual, Garage_Cond) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.02) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


## recipe prop5 - best so far
test5_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_rm(Garage_Qual, Garage_Cond) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.02) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


## recipe prop6
test6_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_rm(Garage_Qual, Garage_Cond) %>%
  step_log(Gr_Liv_Area, First_Flr_SF, base = 10) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(Garage_Type, Condition_1,
             MS_SubClass, Bsmt_Cond,threshold = 0.02) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

## simple 1
simp1_rec <- recipe(Sale_Price ~ Gr_Liv_Area, data = ames_train)

## simple 2
simp2_rec <- 
    recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF, data = ames_train) %>%
    step_log(Gr_Liv_Area, base = 10)

## simple 3
simp3_rec <- 
  recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF + Overall_Qual, 
         data = ames_train) %>%
  step_poly(Gr_Liv_Area, Second_Flr_SF, degree = 2) %>%
  step_dummy(all_nominal_predictors())

## simple 4
simp4_rec <- 
  recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF + Overall_Qual + Total_Bsmt_SF, 
         data = ames_train) %>%
  step_poly(Gr_Liv_Area, Second_Flr_SF, degree = 2) %>%
  step_dummy(all_nominal_predictors())


## simple 5
simp5_rec <- 
  recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF + Overall_Qual + 
           Total_Bsmt_SF + Neighborhood, 
         data = ames_train) %>%
  step_poly(Gr_Liv_Area, Second_Flr_SF, degree = 2) %>%
  step_dummy(all_nominal_predictors())


## simple 6
simp6_rec <- 
  recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF + Overall_Qual + 
           Total_Bsmt_SF + Neighborhood, 
         data = ames_train) %>%
  step_poly(Gr_Liv_Area, Second_Flr_SF, degree = 2) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())


## simple 7
simp7_rec <- 
  recipe(Sale_Price ~ Gr_Liv_Area + Second_Flr_SF + Overall_Qual + 
           Total_Bsmt_SF + Neighborhood + Garage_Cars, 
         data = ames_train) %>%
  step_poly(Gr_Liv_Area, Second_Flr_SF, degree = 2) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())




add_rec <- function(model_rec) {
  
  workflow() %>%
    add_model(lr_spec) %>%
    add_recipe(model_rec)
  
}

models <- list('base linear' = lr_rec,
               'recipe 5' = test5_rec,
               'simple 1' = simp1_rec,
               'simple 2' = simp2_rec,
               'simple 3' = simp3_rec,
               'simple 4' = simp4_rec,
               'simple 5' = simp5_rec,
               'simple 6' = simp6_rec,
               'simple 7' = simp7_rec) %>%
    purrr::map(~add_rec(.)) %>%
    purrr::map(fit, data = ames_train)


# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
  group_by(model) %>%
  ames_met(truth = Sale_Price, estimate = .pred)


test_wf <- workflow() %>%
    add_model(lr_spec) %>%
    add_recipe(test2_rec)

lr_cv <- fit_resamples(test_wf, resamples = ames_fold, metrics = ames_met)

collect_metrics(lr_cv)




# Ridge Regression --------------------------------------------------------

## Model spec
ridge_spec <- 
    linear_reg(mixture = 0, penalty = tune()) %>%
    set_mode('regression') %>%
    set_engine('glmnet')

## Recipe
ridge_rec <-
    test5_rec %>%
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


best_ridge %>%
  extract_fit_engine() %>%
  vip::vi(lambda = select_best(ridge_tune, metric = 'rmse')$penalty) %>%
  top_n(n = 25, wt = abs(Importance)) %>%
  ggplot(aes(x = Importance, y = reorder(Variable, Importance), fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = 'Top 25 variables by Importance')


best_ridge %>%
  extract_fit_engine() %>%
  vip::vip(lambda = select_best(ridge_tune, metric = 'rmse')$penalty,
           mapping = aes(fill = Sign))



best_ridge %>%
  extract_fit_engine() %>%
  vip::vip(mapping = aes(fill = Sign))


## Compare performance

# build named list
models <- list('linear regression' = lr_fit,
               'ridge regression'  = best_ridge)

# apply augment function to all data frame in list
purrr::imap_dfr(models, augment, new_data = ames_train, .id = 'model') %>%
    group_by(model) %>%
    ames_met(truth = Sale_Price, estimate = .pred)


ridge_cv <- fit_resamples(best_ridge, resamples = ames_fold, metrics = ames_met)
collect_metrics(ridge_cv)

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


best_lasso %>%
  extract_fit_engine() %>%
  vip::vi(lambda = select_best(lasso_tune, metric = 'rmse')$penalty) %>%
  top_n(n = 25, wt = abs(Importance)) %>%
  ggplot(aes(x = Importance, y = reorder(Variable, Importance), fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(title = 'Top 25 variables by Importance')


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




# Polynomial Regression ---------------------------------------------------

## Model spec
# same as linear model

## Recipe
poly_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
    step_poly(all_numeric_predictors(), degree = 2) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())

## Workflow
poly_wf <- workflow() %>%
    add_model(lr_spec) %>%
    add_recipe(poly_rec)


## Fit
poly_fit <- fit(poly_wf, data = ames_train)

poly_fit %>%
    augment(new_data = ames_train) %>%
    ames_met(truth = Sale_Price, estimate = .pred)



# Random Forest -----------------------------------------------------------

## Model spec
randf_spec <- rand_forest() %>%
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


randf_fit %>% extract_fit_engine() %>% plot()


models <- append(models, list('random forest' = randf_fit))
