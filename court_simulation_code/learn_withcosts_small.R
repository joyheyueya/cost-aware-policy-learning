get_one_row_that_matches_value <- function(pop, colname, value) {
  pop %>% 
    filter({{colname}} == value) %>% sample_n(1)
}

get_samples_of_all_values <- function(pop, colname) {
  
  unique_values <- pop %>% distinct({{colname}}) %>% pull()
  
  lapply(unique_values, get_one_row_that_matches_value, pop = pop, colname = {{colname}}) %>% 
    bind_rows()
  
}

get_warmup_pop <- function(sim_params) {
  warmup_pop_universe <- generate_samples(sim_params, 1000)
  warmup_pop <- bind_rows(get_samples_of_all_values(warmup_pop_universe, targetgroup),
                          # get_samples_of_all_values(warmup_pop_universe, is_felony),
                          get_samples_of_all_values(warmup_pop_universe, is_male)) %>% 
    sample_n(n()) %>% # Shuffle the samples
    mutate(person_ix = row_number(),
           is_warmup = TRUE,
           control_policy_treat_prob = if_else(person_ix > 2, 1, 0), # Give everyone control except
           transit_policy_treat_prob = if_else(person_ix == 1, 1, 0), # Give first person transit
           rideshare_policy_treat_prob = if_else(person_ix == 2, 1, 0)) %>% # Give second person a ride
    make_decision() %>% 
    mutate(person_ix = person_ix - n())  
}

calc_inferred_appear_prob_thompson <- function(train_pop, batch_pop, num_posteriors) { 
  glm_model <- glm_from_observations(train_pop) 
  fit_model <- get_posteriors_fit(glm_model, num_posteriors)
  calc_inference_func <- calc_inferences(fit_model)
  batch_pop <- batch_pop %>%
    calc_inference_func() %>% 
    calc_reward("(.*_inferred)_appear_prob")
}


get_mle_coef <- function(train_pop) {  
  glm_model <- glm_from_observations(train_pop) 
  fit_model <- get_mle_fit(glm_model)
}

calc_inferred_appear_prob_mle <- function(fit_model, test_pop) {  
  calc_inference_func <- calc_inferences(fit_model)

  test_pop_inferred <- test_pop %>% 
    calc_inference_func() %>% 
    calc_reward("(.*_inferred)_appear_prob")
}
