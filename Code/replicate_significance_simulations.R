#replicate stratified T-test significance simulations for UI-TSM paper in R
library(tidyverse)

run_stratified_t_test <- function(population, strata, n, mu_0 = 0){
  strata_names <- unique(strata)
  strata_sizes <- as.numeric(table(strata))
  K <- length(strata_names)
  a <- prop.table(table(strata))
  strata_sample_means <- rep(NA, K)
  strata_sample_vars <- rep(NA, K)
  for(k in 1:K){
    stratum_samples <- sample(population[strata == strata_names[k]], size = n[k], replace = TRUE)
    strata_sample_means[k] <- mean(stratum_samples)
    strata_sample_vars[k] <- var(stratum_samples)
  }
  mean_estimate <- sum(a * strata_sample_means)
  std_error <- sqrt(sum(a^2 * strata_sample_vars / n))
  #deals with edge cases where we get 0/0, which returns NaN. 
  #the nature of the null is such that we should treat this as a p value of 1:
  #intuitively the conclusion would be that the mean is equal to the null mean with complete certainty.
  if(mean_estimate == mu_0 & std_error == 0){
    p_value <- 1
  } else{
    p_value <- 1 - pt(q = (mean_estimate - mu_0) / std_error, df = min(n))
  }
  p_value
}

num_sims <- 1000
alpha <- 0.05
p <- 0.99
mu <- 0.5050505
sigma <- 0.001
#n_grid <- unique(round(exp(seq(log(5), log(200), length.out = 40))))
n_grid <- round(seq(5, 200, length.out = 40))
levels <- rep(0, length(n_grid))
pop <- c(c(0, rep(mu, 99)), c(0, rep(mu, 99)))
strata <- c(rep(1, 100), rep(2, 100))

for(i in 1:length(n_grid)){
  n <- n_grid[i]
  p_vals <- replicate(num_sims, run_stratified_t_test(pop, strata, n = rep(n, 2), mu_0 = 0.5))
  levels[i] <- mean(p_vals < alpha)
}

sim_data <- data.frame(n = n_grid, t_test_level = levels)

ggplot(sim_data, aes(x = n, y = t_test_level)) +
  geom_line(size = 1.5) +
  geom_hline(yintercept = 0.05, linetype = 'dashed') +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) +
  ylim(0,1) +
  ylab("Estimated significance level") +
  xlab("Sample size in each stratum")


#T-scores with varying numbers of zeros
#125 samples 1 zero
sam_125 <- c(c(rep(mu, 124), 0), rep(mu, 125))
sam_135 <- c(c(rep(mu, 134), 0), rep(mu, 135))
SE_125 <- 1/2 * sqrt((var(sam_125[1:125]) + var(sam_125[126:250])) / 125)
SE_135 <- 1/2 * sqrt((var(sam_135[1:135]) + var(sam_135[136:270])) / 135)
T_125 <- (mean(sam_125) - 0.5) / SE_125
T_135 <- (mean(sam_135) - 0.5) / SE_135
P_125 <- 1 - pt(T_125, df = 125 - 1)
P_135 <- 1 - pt(T_135, df = 135 - 1)



