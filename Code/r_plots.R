#plots for AoAS paper on SFSNP stratified inference
library(tidyverse)
library(latex2exp)
library(xtable)


###### significance levels at different sample sizes for stratified mixture distributions ######3
significance_data <- read_csv("significance_simulation_results.csv")

ggplot(significance_data, aes(x = n, y = t_test_level)) +
  geom_line(size = 2) +
  geom_hline(yintercept = 0.05, linetype = 'dashed') +
  theme_bw() +
  theme(
    text = element_text(size = 24), 
    axis.text = element_text(size = 18),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 18)) +
  ylim(0,1) +
  ylab("Estimated significance level") +
  xlab("Sample size in each stratum")

####### stopping times for specific intersection nulls ############
intersection_null_data <- read_csv("intersection_null_stopping_times.csv") %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "fixed" ~ "Fixed", "smooth_predictable" ~ "Smooth Adaptive")) %>%
  #filter(allocation == "round_robin") %>%
  #filter(method == "product") %>%
  #filter(alt == 0.55) %>%
  filter(delta %in% c(0.5)) %>%
  filter(bet %in% c("Fixed", "AGRAPA")) %>%
  mutate(eta = ifelse(eta %in% c("[0, 1]","[1, 0]","[0.5, 0.5]"), eta, "proj"))


#martingale sizes
ggplot(intersection_null_data, aes(x = time, y = mart, color = eta)) +
  geom_line() +
  facet_grid(bet ~ allocation)
#sample sizes
ggplot(intersection_null_data, aes(x = time, y = n_1, color = eta, linetype = eta)) +
  geom_line(alpha = 0.7, size = 1.2) +
  facet_grid(bet ~ allocation)

#stopping time plot
ggplot(intersection_null_data, aes(x = alt, y = stopping_time, color = eta)) +
  geom_line() +
  facet_grid(bet ~ allocation)

####### stopping times for point mass distributions #######
point_mass_stopping_times <- read_csv("point_mass_results.csv") %>%
  filter(method != "uinnsm_fisher") %>%
  filter(delta != 0.1) %>%
  filter(allocation != "larger_means") %>%
  filter(bet != "smooth_predictable") %>%
  mutate(method = case_match(method, "lcb" ~ "LCB", "uinnsm_fisher" ~ "UI-TS Fisher",  "uinnsm_product" ~ "UI-TS")) %>%
  mutate(method = factor(method, levels = c("UI-TS", "LCB"))) %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "fixed_predictable" ~ "Predictable plug-in", "inverse" ~ "Inverse", "smooth_predictable" ~ "Negative exponential")) %>%
  mutate(allocation = case_match(allocation,"greedy_kelly" ~ "Greedy",  "predictable_kelly" ~ "Predictable Kelly", "round_robin" ~ "Round robin", "larger_means" ~ "Larger means")) %>%
  mutate(sample_size = ifelse(allocation == "Greedy", stopping_time, sample_size)) %>%
  mutate(delta_long = paste("Stratum gap =", delta))



#Kelly optimal stopping time for point mass populations
#add lower bound to stopping time plots
obj <- function(eta, mu_1, mu_2){
  eta/(1-eta) - (mu_1/mu_2)
}
get_eta_star <- Vectorize(function(mu_1, mu_2){
  uniroot(obj, mu_1 = mu_1, mu_2 = mu_2, lower = 1e-5, upper = 1 - 1e-5)$root
}, c("mu_1", "mu_2"))
alpha <- 0.05
kelly_optimal_stopping_times <- point_mass_stopping_times %>%
  #filter(method == "UI-TSM Product", bet == "Fixed", allocation == "Round Robin") %>%
  mutate(mu_1 = alt - delta/2, mu_2 = alt + delta/2) %>%
  mutate(eta_star = get_eta_star(mu_1, mu_2)) %>%
  mutate(kost = ceiling(log(alpha) / (log(eta_star) - log(mu_1))))



ggplot(point_mass_stopping_times %>% filter(n_bands == 100), aes(x = alt, y = sample_size, color = bet, linetype = method)) +
  geom_line(size = 1.5) +
  #geom_line(data = kelly_optimal_stopping_times, aes(y = kost), linetype = 'dashed', color = 'black') +
  facet_grid(allocation ~ delta_long) +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14), 
    panel.spacing.x = unit(6, "mm")) +
  scale_linetype_manual(values = c("solid","dashed","dotted")) +
  ylab("Sample Size") +
  xlab("Global Mean") +
  labs(colour = "Selection rule", linetype = "Method") +
  scale_y_log10() +
  scale_x_continuous(breaks = seq(0.5,0.75,by=0.05))

#how things are effected by the number of points 
G_stopping_time_table <- point_mass_stopping_times %>% 
  filter(method == "UI-TS Product", bet == "AGRAPA") %>%
  group_by(n_bands) %>%
  summarize(mean_stop = mean(stopping_time), mean_run_time = mean(run_time))
xtable(G_stopping_time_table)



###### bernoulli stopping times #####
full_data <- read_csv("all_bernoulli_simulations.csv")
mean_data <- full_data %>%
  select(-...1) %>%
  filter(!(method %in% c("uitsm_uniform", "uitsm_vertex", "unstrat_fixed"))) %>%
  filter(!((method == "unstrat_agrapa") )) %>%
  group_by(alt, delta, method, bet, allocation) %>%
  summarize(
    expected_stopping_time = mean(stopping_time), 
    prob_stop = mean(stopping_time < 4199),
    sd_stopping_time = sd(stopping_time), 
    min_stopping_time = min(stopping_time), 
    max_stopping_time = max(stopping_time),
    expected_sample_size = mean(sample_size), 
    prob_stop = mean(stopping_time < 4199),
    sd_sample_size = sd(sample_size), 
    min_sample_size = min(sample_size), 
    max_sample_size = max(sample_size)) %>%
  filter(bet != "smooth_predictable") %>%
  mutate(method = case_match(method, "lcb" ~ "LCB", "uinnsm_fisher" ~ "UI-TS Fisher",  "uinnsm_product" ~ "UI-TS")) %>%
  mutate(method = factor(method, levels = c("UI-TS", "UI-TS Fisher", "LCB"))) %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "inverse" ~ "Inverse", "apriori_bernoulli" ~ "Kelly optimal", "bernoulli" ~ "ALPHA-ST", "fixed_plugin" ~ "Predictable plug-in", "smooth_predictable" ~ "Negative exponential")) %>%
  mutate(allocation = case_match(allocation, "predictable_kelly" ~ "Predictable Kelly", "round_robin" ~ "Round Robin", "greedy_kelly" ~ "Greedy")) %>%
  filter(!is.na(expected_stopping_time)) %>%
  #filter(delta == 0.5) %>%
  pivot_longer(cols = c("expected_sample_size", "prob_stop"), names_to = "Measure") %>%
  mutate(Measure = case_match(Measure, "prob_stop" ~ "Probability of Stopping", "expected_sample_size" ~ "Expected Sample Size")) %>%
  mutate(delta_long = paste("Stratum gap =", delta))  %>%
  filter(alt != 0.75)


ggplot(mean_data %>% filter(Measure == "Expected Sample Size", allocation == "Round Robin"), aes(x = alt, y = value, color = bet, linetype = method)) +
  geom_line(size = 1.5) +
  theme_bw() + 
  ylab("Expected Sample Size") +
  geom_text(
    data = data.frame(x = 0.68, y = 1400, text = "(A)", method = NA), 
    aes(x = x, y = y, label = text), 
    color = "black",
    size = 16) +
  xlab("Global Mean") +
  labs(colour = "Betting Rule", linetype = "Method") +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) +
  scale_y_log10()

ggplot(mean_data %>% filter(Measure == "Probability of Stopping", allocation == "Round Robin"), aes(x = alt, y = value, color = bet, linetype = method)) +
  ylab(TeX("$P(\\tau < 4200)$")) +
  geom_text(
    data = data.frame(x = 0.68, y = .8, text = "(B)", method = NA), 
    aes(x = x, y = y, label = text), 
    color = "black",
    size = 16) +
  xlab(TeX("Global Mean ($\\mu$)")) +
  labs(colour = "Betting Rule", linetype = "Method") +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) +
  geom_line(size = 1.5)


ggplot(mean_data %>% filter )

ggplot(mean_data, aes(x = alt, y = value, color = bet, linetype = method)) +
  ylab("Measure") +
  facet_grid(.~ Measure, scales = "free_y") +
  xlab(TeX("Global Mean ($\\mu$)")) +
  labs(colour = "Betting Rule", linetype = "Method") +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) +
  geom_line(size = 1.5)







###### gaussian stopping times #######
full_data <- read_csv("all_gaussian_simulations.csv")
mean_data <- full_data %>%
  select(-...1) %>%
  group_by(across(-c(rep, stopping_time, sample_size))) %>%
  summarize(
    expected_stopping_time = mean(stopping_time), 
    expected_sample_size = mean(sample_size),
    prob_stop = mean(stopping_time < 1000),
    sd_stopping_time = sd(stopping_time), 
    min_stopping_time = min(stopping_time), 
    max_stopping_time = max(stopping_time)) %>%
  filter(bet != "smooth_predictable") %>%
  mutate(method = case_match(method, "lcb" ~ "LCB",  "uits" ~ "UI-TS", "unstrat" ~ "Unstratified TSM")) %>%
  mutate(method = factor(method, levels = c("UI-TS", "LCB", "Unstratified TSM"))) %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "inverse" ~ "Inverse", "fixed_predictable" ~ "Predictable plug-in", "smooth_predictable" ~ "Negative Exponential")) %>%
  mutate(allocation = case_match(allocation, "predictable_kelly" ~ "Predictable Kelly", "round_robin" ~ "Round Robin", "greedy_kelly" ~ "Greedy")) %>%
  mutate(delta = paste("Stratum gap =", delta)) %>%
  mutate(sd = paste("SD =", sd)) %>%
  filter(K == 2) %>%
  mutate(K = paste("K =", K)) %>%
  #filter(allocation_rule == "Round Robin" | ((allocation_rule == "Neyman") & (method == "Unstratified AGRAPA"))) %>%
  rename(Method = method) %>%
  filter(allocation == "Round Robin") 


expected_stop_plot <- ggplot(mean_data, aes(x = global_mean, y = expected_sample_size, color = bet, linetype = Method)) +
  geom_line(size = 1) +
  facet_grid(sd ~ delta) +
  xlab("Global Mean") +
  ylab("Expected Sample Size") +
  labs(colour = "Betting rule", linetype = "Method") + 
  theme_bw() +
  theme(text = element_text(size = 18), axis.text = element_text(size = 14)) +
  scale_y_log10() 

expected_stop_plot
  


######### gradient descent run times ########
runtime_data <- read_csv("pgd_run_times_short.csv")

runtime_table <- runtime_data %>% 
  select(N_k, K, run_time, method, sample_size) %>% 
  mutate(N = N_k * K) %>% 
  pivot_wider(names_from = "method", values_from = c("run_time", "sample_size")) %>% 
  filter(N_k %in% c(10, 100, 1000)) %>%
  mutate(N_k = as.integer(N_k), 
         N = as.integer(N), 
         K = as.integer(K), 
         sample_size_lcb = as.integer(sample_size_lcb),
         sample_size_uits = as.integer(sample_size_uits))

print(xtable(runtime_table, digits = 1), include.rownames = F)

ggplot(runtime_data, aes(x = K, y = run_time, color = as_factor(N_k), linetype = method)) +
  geom_line() 


######## kelly-optimal UI-TS and betting TSM stopping time, pointmasses #######
alpha <- 0.05
#get the minimizer of the kelly-optimal UI-TS
obj <- function(eta, mu_1, mu_2){
  eta/(1-eta) - (mu_1/mu_2)
}
get_eta_star <- Vectorize(function(mu_1, mu_2){
  uniroot(obj, mu_1 = mu_1, mu_2 = mu_2, lower = 1e-5, upper = 1 - 1e-5)$root
}, c("mu_1", "mu_2"))
#get the stopping time of the kelly-optimal (unstratified) betting TSM
get_Etau_unstratified <- Vectorize(function(mu_1, mu_2){
  if(mu_1 + mu_2 < 1/2){
    stop("null is true!")
  }
  l_star <- ifelse((mu_1 >= 1/2) & (mu_2 >= 1/2), 2,
                        (1-mu_1-mu_2) / (2 * (mu_1 - 1/2) * (mu_2 - 1/2)))
  l_star <- pmin(l_star, 2)
  delta_B_star <- (1/2) * log(1 + l_star * (mu_1 - 1/2)) + (1/2) * log(1 + l_star * (mu_2 - 1/2))
  tau_star <- -log(alpha) / delta_B_star
  tau_star
}, c("mu_1", "mu_2"))



mu_1 <- seq(0,1,by=.001)
mu_2 <- seq(0,1,by=.001)
alt_grid <- expand.grid(mu_1, mu_2) %>%
  rename("mu_1" = Var1, "mu_2" = Var2) %>%
  filter((mu_1/2 + mu_2/2) > (1/2 + 0.005)) %>%
  mutate(eta_star = get_eta_star(mu_1, mu_2)) %>%
  mutate(kost = ceiling(log(alpha) / (log(eta_star) - log(mu_1))))
alt_grid_unstratified <- expand.grid(mu_1, mu_2) %>%
  rename("mu_1" = Var1, "mu_2" = Var2) %>%
  filter((mu_1/2 + mu_2/2) > (1/2 + 0.005)) %>%
  mutate(kost = ceiling(get_Etau_unstratified(mu_1, mu_2))) 

rel_eff_grid <- alt_grid %>%
  left_join(alt_grid_unstratified, by = c("mu_1", "mu_2")) %>%
  mutate(relative_efficiency = kost.y / kost.x) %>%
  mutate(mu = (mu_1 + mu_2) / 2) %>%
  filter(mu %in% c(0.51, 0.52, 0.55, 0.6, 0.75), mu_1 > mu_2) %>%
  mutate(stratum_diff = mu_1 - mu_2) %>%
  mutate(mu = as_factor(mu))



ggplot(alt_grid, aes(x = mu_1, y = mu_2, fill = kost)) +
  geom_raster() +
  xlim(0,1) +
  ylim(0,1) +
  geom_abline(slope = -1, intercept = 1, linetype = 'dashed') +
  scale_fill_viridis_c(
    name = TeX("$\\log_{10}(\\tau^*)$"), 
    trans = "log10", 
    option = "turbo", 
    limits = c(5, 1e3),
    oob = scales::squish) +
  theme_bw() +
  geom_text(data = data.frame(x = 0.065, y = 0.8, text = "(A)", kost = NA), aes(x = x, y = y, label = text), size = 16) +
  theme(text = element_text(size = 18), axis.text = element_text(size = 14)) +
  xlab(TeX("$\\mu_1$")) +
  ylab(TeX("$\\mu_2$")) 

ggplot(alt_grid_unstratified, aes(x = mu_1, y = mu_2, fill = kost)) +
  geom_raster() +
  xlim(0,1) +
  ylim(0,1) +
  geom_abline(slope = -1, intercept = 1, linetype = 'dashed') +
  scale_fill_viridis_c(
    name = TeX("$\\log_{10}(\\tau^*)$"), 
    trans = "log10", 
    option = "turbo", 
    limits = c(5, 1e3), 
    oob = scales::squish) +
  geom_text(data = data.frame(x = 0.065, y = 0.8, text = "(B)", kost = NA), aes(x = x, y = y, label = text), size = 16) +
  theme_bw() +
  theme(text = element_text(size = 18), axis.text = element_text(size = 14)) +
  xlab(TeX("$\\mu_1$")) +
  ylab(TeX("$\\mu_2$")) 

ggplot(rel_eff_grid, aes(x = stratum_diff, y = relative_efficiency, color = mu)) +
  geom_hline(yintercept = 1, linetype = 'dashed') +
  geom_hline(yintercept = 2, linetype = 'dotted') +
  geom_line(size = 2) +
  coord_cartesian(xlim = c(0, 0.4), ylim = c(0,10)) +
  geom_text(
    data = data.frame(x = 0.065, y = 8, text = "(C)", mu = NA), 
    aes(x = x, y = y, label = text), 
    color = "black",
    size = 16) +
  scale_color_viridis_d(option = "turbo") +
  theme_bw() +
  theme(text = element_text(size = 18), axis.text = element_text(size = 14)) +
  xlab(TeX("$\\mu_1 - \\mu_2$")) +
  ylab("Stopping time ratio (unstratified over stratified)") +
  labs(color = TeX("$\\mu = (\\mu_1 + \\mu_2)/2"))


########### negative exponential bets as function of eta ############
eta <- seq(0.05,1,by=.001)
lam_lim <- 1/eta
mu <- 0.8
T_k <- 10
Xbar <- mu # this is the sample mean at time t
SDbar <- sqrt(mu * (1-mu))
st_bets <- pmax(0, (Xbar / eta - 1) / (1 - eta)) #this is the KO bet using the sample mean as the estimate
agrapa_bets <- pmin(pmax(0, (Xbar - eta) / (SDbar^2 + (Xbar-eta)^2)), 1/eta) 
student_bets <- pmin(pmax(0, (Xbar - eta) / (SDbar^2)), 1/eta) 
#b <- 1
#a <- log(b + 1)
eps <- 0.9 - SDbar
predictable_plugin <- sqrt(2 * log(2/0.05) / (T_k * log(T_k + 1)))
ne_bets <- exp(1 - ((1 - log(eps)) / Xbar) * eta)
inverse_bets <- 0.5 / eta
#ne_bets <- exp(1 - b * eta)
convex_bets <- exp(-eta)
#inverse_squared_bets <- 1 / (1 + eta)^2

bets_frame <- data.frame(
  eta = eta, 
  maximal_bets = lam_lim, 
  st_bets = st_bets,
  predictable_plugin,
  inverse_bets = inverse_bets,
  agrapa_bets = agrapa_bets) %>%
  pivot_longer(cols = c("st_bets", "inverse_bets", "predictable_plugin", "maximal_bets", "agrapa_bets"), names_to = "Bet") %>%
  mutate(Bet = case_match(Bet, "st_bets" ~ "Kelly optimal", "inverse_squared_bets" ~ "Inverse squared", "inverse_bets" ~ "Inverse (c = 0.5)", "agrapa_bets" ~ "AGRAPA", "maximal_bets" ~ "Upper bound", "student_bets" ~ "Studentized", "predictable_plugin" ~ "Predictable plug-in"))
ggplot(bets_frame, aes(x = eta, y = value, color = Bet)) +
  geom_vline(xintercept = Xbar, linetype = 'dashed') +
  geom_line(linewidth = 1.2) +
  #scale_color_viridis_d(option = "viridis") +
  coord_cartesian(ylim = c(0,5)) +
  ylab(TeX("$\\lambda(\\eta)$")) +
  xlab(TeX("$\\eta$")) +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) 

c <- 0.75
plot(log(c) - log(eta) / (1 - eta))



########### cutoffs for different combining functions #########
alpha <- seq(.001,0.5,by=.001)

cutoff_phi_F_2 <- qchisq(1-alpha, df = 4)
cutoff_phi_F_3 <- qchisq(1-alpha, df = 6)
cutoff_phi_F_4 <- qchisq(1-alpha, df = 8)
cutoff_phi_F_5 <- qchisq(1-alpha, df = 10)
cutoff_phi_P <- -2 * log(alpha)


plot(cutoff_phi_F_5 ~ alpha, type = 'l', col = 'darkorange3', lwd = 2, lty = 'dashed', ylim = c(0,30), ylab = "Cutoff", main = "Fisher vs Product Combining", cex.main = 1.5, cex.axis = 1.5, cex.lab = 1.5)
points(cutoff_phi_F_4 ~ alpha, type = 'l', col = 'violet', lwd = 2, lty = 'dotted')
points(cutoff_phi_F_3 ~ alpha, type = 'l', col = 'goldenrod', lwd = 2, lty = 'longdash')
points(cutoff_phi_F_2 ~ alpha, type = 'l', col = 'forestgreen', lwd = 2, lty = 'twodash')
points(cutoff_phi_P ~ alpha, type = 'l', col = 'steelblue', lwd = 2)
legend(x = 0.3, y = 25, legend = c("phi_F, K = 5", 
                                   "phi_F, K = 4", 
                                   "phi_F, K = 3",
                                   "phi_F, K = 2",
                                   "phi_P"),
       col = c("darkorange3", "violet", "goldenrod", "forestgreen", "steelblue"),
       lwd = 2,
       lty = c("dashed", "dotted", "longdash", "twodash", "solid")
)

#Tippett's combining function vs maximization
cutoff_phi_T_2 <- qbeta(alpha, shape1 = 1, shape2 = 2)
cutoff_phi_T_3 <- qbeta(alpha, shape1 = 1, shape2 = 3)
cutoff_phi_T_4 <- qbeta(alpha, shape1 = 1, shape2 = 4)
cutoff_phi_T_5 <- qbeta(alpha, shape1 = 1, shape2 = 5)
cutoff_phi_M <- alpha


plot(cutoff_phi_T_5 ~ alpha, type = 'l', col = 'darkorange3', lwd = 2, lty = 'dashed', ylim = c(0,0.5), ylab = "Cutoff",  main = "Tippett vs Maximization Combining", cex.main = 1.5, cex.axis = 1.5, cex.lab = 1.5)
points(cutoff_phi_T_4 ~ alpha, type = 'l', col = 'violet', lwd = 2, lty = 'dotted')
points(cutoff_phi_T_3 ~ alpha, type = 'l', col = 'goldenrod', lwd = 2, lty = 'dashed')
points(cutoff_phi_T_2 ~ alpha, type = 'l', col = 'forestgreen', lwd = 2, lty = 'dashed')
points(cutoff_phi_M ~ alpha, type = 'l', col = 'steelblue', lwd = 2)
legend(x = 0.1, y = .4, legend = c("phi_T, K = 5", 
                                   "phi_T, K = 4", 
                                   "phi_T, K = 3",
                                   "phi_T, K = 2",
                                   "phi_M"),
       col = c("darkorange3", "violet", "goldenrod", "forestgreen", "steelblue"),
       lwd = 2,
       lty = c("dashed", "dotted", "longdash", "twodash", "solid")
)


######### martingale shape under inverse bets #######
eta <- seq(0,1, by = .01)
eps <- 0.9
b <- (1-log(eps)) / 0.5
X_1 <- 0.7
X_2 <- 0.1
lambda <- function(eta){0.5 / eta}
mart <- function(eta){(1 + lambda(eta) * (X_1 - eta)) * (1 + lambda(1-eta) * (X_2 - (1-eta)))}
lmart <- function(eta){log(1 + lambda(eta) * (X_1 - eta)) + log(1 + lambda(1-eta) * (X_2 - (1-eta)))}
plot(mart(eta) ~ eta, type = 'l')




