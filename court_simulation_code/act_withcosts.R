reward_formula <- function(x) {
  # Simply count appearances for now
  # Though this could in theory depend on more complex things
  # Like a per-person bonus for treatment
  x 
}

calc_reward <- function(pop, outcome_col_regex) {
  rewards <- pop %>% 
    select(matches(outcome_col_regex)) %>% 
    mutate_all(~ reward_formula(.)) %>% 
    rename_with(~ str_replace(., outcome_col_regex, "\\1_reward"))
  
  bind_cols(pop, rewards)
}

decision_crosser <- function(pop, decision_col_regex, cross_col_regex, new_col_name) { 
  
  # Pull out decisions
  decisions <- pop %>% 
    select(matches(decision_col_regex)) %>% 
    select(sort(peek_vars()))
  
  # Pull out x per action
  x <- pop %>% 
    select(matches(cross_col_regex, perl = TRUE)) %>% 
    select(sort(peek_vars()))
  
  # Cross decisions with x, and summarize total x
  crossed <- (x * decisions) %>% 
    as_tibble() %>% 
    rename_with(~ str_replace(., cross_col_regex, glue("\\1_{new_col_name}"))) %>% 
    rowwise() %>% 
    mutate({{new_col_name}} := sum(across(contains(new_col_name)))) %>% 
    ungroup()
  
  bind_cols(pop, crossed)
  
}


make_decision <- function(pop) {
  
  arms <- pop %>% 
    select(contains("policy_treat_prob")) %>% 
    select(sort(peek_vars())) 
  
  # Decide on an action for each row
  decision_positions <- arms %>% 
    apply(., 1, rmultinom, n = 1, size = 1) %>%  # Randomly draw action based on probs
    apply(., 2, which.max)  # Get position of action
  
  # Get name of action
  arm_names <- colnames(arms) %>% 
    str_remove("_policy_treat_prob")
  decisions <- arm_names[decision_positions]
  
  # Get all arm columns (to fill with zeros on final output)
  # (Meant to be an empty dataframe)
  all_arm_cols <- arms %>% 
    filter(FALSE) %>% 
    rename_with(~ str_replace(., "policy_treat_prob", "decision"))
  
  # Add one-hot columns for decisions
  one_hot_decisions <- decisions %>% 
    as_tibble() %>% 
    mutate(person_ix = pop$person_ix,
           decision = 1) %>% 
    pivot_wider(id_cols = "person_ix", names_from = value, names_glue = "{value}_decision", values_from = decision) %>% 
    mutate(across(where(is.numeric), ~ replace_na(., 0))) %>% 
    bind_rows(all_arm_cols) %>% 
    mutate(across(contains("_decision"), ~ replace_na(., 0))) %>% 
    select(sort(peek_vars()), -person_ix)
  
  # Join decisions back to pop
  output <- pop %>% 
    bind_cols(one_hot_decisions)
  
  # Get the observed outcome for chosen action(s)
  appearances <- output %>% 
    decision_crosser("_decision", "(.*)_true_appear_outcome", "observed_appear_outcome") %>% 
    select(person_ix, observed_appear_outcome)
  
  # Get the observed reward from the chosen action(s)
  rewards <- output %>% 
    calc_reward("(.*_true)_appear_outcome") %>% 
    decision_crosser("_decision", "(.*)_true_reward", "observed_reward") %>% 
    select(person_ix, observed_reward)
  
  # Get the observed spending from the chosen action(s)
  spending <- output %>% 
    decision_crosser("_decision", "(.*)_cost", "observed_spending") %>% 
    select(person_ix, observed_spending)

  # Output: pop with decisions, observed outcomes, and spending
  output %>% 
    mutate(decision = decisions) %>% 
    left_join(appearances, by = "person_ix") %>% 
    left_join(rewards, by = "person_ix") %>% 
    left_join(spending, by = "person_ix") 
    
}
