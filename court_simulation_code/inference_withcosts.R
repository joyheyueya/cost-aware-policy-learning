glm_from_observations <- function(pop) {
  
  pop <- pop %>% 
    rename_with(~ str_remove(., "_decision"))
  
  glm(observed_appear_outcome ~ 1 + 
        targetgroup + 
        is_felony +
        is_male +
        age + 
        log_distance + 
        fta_hist +
        appt_hist_inv +
        fta_hist : appt_hist_inv +
        rideshare + 
        transit : log_distance,
      data = pop, family = "binomial")
  
}

prep_lm_output <- function(lm_tibble) {
  
  lm_tibble %>% 
    rename_all(~ str_remove(., "TRUE")) %>% 
    rename_all(~ make_clean_names(.)) %>% 
    rename_all(~ glue("est_{.}_coef")) %>% 
    mutate(model_ix = factor(row_number())) %>% 
    select(model_ix, everything()) %>% 
    mutate(across(where(is.numeric), ~ replace_na(., 0))) # Allow predictions with incomplete models
}

get_mle_fit <- function(lm_obj) {
  
  fit_mle <- lm_obj %>% 
    tidy() %>% 
    select(term, estimate) %>% 
    pivot_wider(names_from = term, values_from = estimate) %>% 
    prep_lm_output()
  
}


get_posteriors_fit <- function(lm_obj, num_posteriors) {
  
  # Need to add 1 because sim returns a default model (intercept = 1, coefs = 0) 
  # if you only ask for 1 posterior
  # (Seems to be a bug (or a conceptual misunderstanding) with `sim`)
  if (num_posteriors == 1) {
    num_posteriors <- 2
    remove_posterior <- TRUE
  } else {
    remove_posterior <- FALSE
  }
  
  fit_posterior <- sim(lm_obj, n.sims = num_posteriors) %>% 
    coef() %>% 
    as_tibble() %>% 
    prep_lm_output()
  
  if (remove_posterior) {
    fit_posterior <- fit_posterior %>% 
      slice(1)
  }
  
  fit_posterior
  
}

calc_inferences <- function(model_coefs) {
  
  function(pop) {
    # I found it necessary to do it this way to preserve passing multiple models 
    # (e.g., 25 posterior models at once)
    
    pop <- pop %>% 
      crossing(model_coefs)
    
    pop %>% 
      mutate(control_inferred_appear_prob   = response_model(intercept = est_intercept_coef,
                                                             targetgroup = targetgroup,
                                                             targetgroup_coef = est_targetgroup_coef,
                                                             is_felony = is_felony,
                                                             is_felony_coef = est_is_felony_coef,
                                                             is_male = is_male,
                                                             is_male_coef = est_is_male_coef,
                                                             age = age,
                                                             age_coef = est_age_coef,
                                                             log_distance = log_distance,
                                                             log_distance_coef = est_log_distance_coef,
                                                             fta_hist = fta_hist,
                                                             fta_hist_coef = est_fta_hist_coef,
                                                             appt_hist_inv = appt_hist_inv,
                                                             appt_hist_inv_coef = est_appt_hist_inv_coef,
                                                             fta_hist_appt_hist_inv_coef = est_fta_hist_appt_hist_inv_coef,
                                                             rideshare = 0,
                                                             rideshare_coef = est_rideshare_coef,
                                                             transit = 0,
                                                             transit_log_distance_coef = est_log_distance_transit_coef),
             transit_inferred_appear_prob   = treat_response_model(control_appear_prob = control_inferred_appear_prob,
                                                                   rideshare = 0,
                                                                   rideshare_coef = est_rideshare_coef,
                                                                   transit = 1,
                                                                   log_distance = log_distance,
                                                                   transit_log_distance_coef = est_log_distance_transit_coef),
             rideshare_inferred_appear_prob = treat_response_model(control_appear_prob = control_inferred_appear_prob,
                                                                   rideshare = 1,
                                                                   rideshare_coef = est_rideshare_coef,
                                                                   transit = 0,
                                                                   log_distance = log_distance,
                                                                   transit_log_distance_coef = est_log_distance_transit_coef))
  } 
  
}

calc_inferences_UCB <- function(model_coefs, percentile = 0.975) {
  
  function(pop) {
    
    optimistic_appear_probs <- pop %>%
      calc_inferences(model_coefs)() %>%
      group_by(person_ix) %>%
      summarize(across(matches(".*inferred_appear_prob"), ~ quantile(., percentile)),
                .groups = "drop")
    
    pop %>%
      left_join(optimistic_appear_probs, by = "person_ix") %>%
      mutate(model_ix = as.factor(1))
    
  }
  
}

calc_inferences_oracle <- function(pop) {
  # This oracle can see probabilities of outcomes
  pop %>%
    mutate(across(matches("true_appear_prob"), ~ ., .names = "{.col}_inferred"),
           model_ix = as_factor(1)) %>%
    rename_with(~ str_replace(., "true_appear_prob_inferred", "inferred_appear_prob"))

  # # This oracle can see actual outcomes
  # pop %>%
  #   mutate(across(matches("true_appear_outcome"), ~ ., .names = "{.col}_inferred"),
  #          model_ix = as_factor(1)) %>%
  #   rename_with(~ str_replace(., "true_appear_outcome_inferred", "inferred_appear_prob")) 
}


