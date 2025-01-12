require(arm)
require(broom)
require(janitor)
require(tidyverse)
require(tidyselect)
require(glue)
require(scales)
require(parallel)
require(pracma)
require(dplyr)
require(tidyr)


empirical_sample <- read_csv("court.csv") 

logit_func <- function(x) {
  log(x / (1-x))
}

inv_logit_func <- function(x) {
  1 / (1 + exp(-x))
}

response_model <- function(intercept,
                           targetgroup,
                           targetgroup_coef,
                           is_felony,
                           is_felony_coef,
                           is_male,
                           is_male_coef,
                           age,
                           age_coef,
                           log_distance,
                           log_distance_coef,
                           fta_hist,
                           fta_hist_coef,
                           appt_hist_inv,
                           appt_hist_inv_coef,
                           fta_hist_appt_hist_inv_coef,
                           rideshare,
                           rideshare_coef,
                           transit,
                           transit_log_distance_coef) {
  
  control_Z <- intercept +
               targetgroup * targetgroup_coef +
               is_felony * is_felony_coef + 
               is_male * is_male_coef + 
               age * age_coef +
               log_distance * log_distance_coef +
               fta_hist * fta_hist_coef + 
               appt_hist_inv * appt_hist_inv_coef +
               fta_hist * appt_hist_inv * fta_hist_appt_hist_inv_coef
  
  treat_response_model(inv_logit_func(control_Z),
                       rideshare,
                       rideshare_coef,
                       transit,
                       log_distance,
                       transit_log_distance_coef)
                       
}

treat_response_model <- function(control_appear_prob,
                                 rideshare,
                                 rideshare_coef,
                                 transit, 
                                 log_distance,
                                 transit_log_distance_coef) {
  
  control_appear_logit <- logit_func(control_appear_prob)

  Z <- control_appear_logit +
       replace_na(rideshare * rideshare_coef, 0) + 
       transit * log_distance * transit_log_distance_coef

  inv_logit_func(Z)
  
}

pop_sample <- function(num_sim,
                       param_rideshare_coef,
                       param_transit_log_distance_coef,
                       param_rideshare_distance_price_permi,
                       param_transit_price_constant,
                       param_max_distance) {
  
  response_prob <- function(control_appear_prob, rideshare, transit, log_distance) {
    treat_response_model(control_appear_prob, 
                         rideshare, param_rideshare_coef,
                         transit, log_distance, 
                         param_transit_log_distance_coef)
  }
  
  response <- function(prob, u_r) {
    as.integer(u_r <= prob)
  }
  
  empirical_sample %>% 
    sample_n(num_sim, replace = TRUE) %>% 
    mutate(transit_true_appear_prob  =  response_prob(control_true_appear_prob,
                                                      rideshare = 0, transit = 1, 
                                                      log_distance),
           rideshare_true_appear_prob = response_prob(control_true_appear_prob,
                                                      rideshare = 1, transit = 0, 
                                                      log_distance),
           u_r = runif(n()),
           control_true_appear_outcome = response(control_true_appear_prob, u_r),
           transit_true_appear_outcome = response(transit_true_appear_prob, u_r),
           rideshare_true_appear_outcome = response(rideshare_true_appear_prob, u_r),
           control_cost = 0.1,
           transit_cost = param_transit_price_constant,
           rideshare_cost = exp(log_distance + log(param_max_distance)) * param_rideshare_distance_price_permi)
  
}
                                                   

generate_samples <- function(params, n) {
  
  param_names <- names(params)
  
  do.call(pop_sample, c(params, num_sim = n)) %>% 
    mutate(person_ix = row_number()) %>% 
    select(person_ix, everything())
  
}
