#bits and bobs for AoAS paper on SFSNP stratified inference
library(tidyverse)

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


####### stopping times for small point mass distributions #######
point_mass_data <- read_csv("point_mass_results.csv") %>%
  mutate(method = case_match(method, "lcb" ~ "LCB", "uinnsm" ~ "UI-NNSM")) %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "fixed" ~ "Fixed", "smooth_predictable" ~ "Smooth Adaptive")) %>%
  rename(Method = method, `Betting Strategy` = bet)


ggplot(point_mass_data, aes(x = alt, y = stopping_time, color = Method, linetype = allocation)) +
  geom_line(size = 1.5) +
  facet_grid(`Betting Strategy` ~ delta) +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14),
    legend.key.width = unit(2, "cm"),
    legend.text = element_text(size = 14)) +
  scale_linetype_manual(values = c("solid","dashed","dotted")) +
  ylab("Stopping Time") +
  xlab("Reported Assorter Mean") +
  scale_y_log10()



####### stopping times for comparison audits #######
comparison_audit_data <- read_csv("comparison_audit_results.csv") %>%
  # pivot_longer(
  #   cols = c("stopping_time_uinnsm", "stopping_time_lcb"), 
  #   names_to = "method", 
  #   values_to = "stopping_time",
  #   names_prefix = "stopping_time_") %>% 
  filter(stratum_gap == 0.5) %>%
  mutate(method = case_match(method, "lcbs" ~ "LCB", "ui-nnsm" ~ "UI-NNSM")) %>%
  mutate(bet = case_match(bet, "agrapa" ~ "AGRAPA", "fixed" ~ "Fixed", "smooth_predictable" ~ "Smooth Adaptive")) %>%
  rename(Method = method, `Betting Strategy` = bet)
  

ggplot(comparison_audit_data, aes(x = A_c, y = stopping_time, color = Method, linetype = `Betting Strategy`)) +
  geom_line(size = 1.5) +
  facet_grid(~ allocation) +
  theme_bw() +
  theme(
    text = element_text(size = 18), 
    axis.text = element_text(size = 14), 
    legend.text = element_text(size = 14),
    legend.key.width = unit(2, "cm")) +
  ylab("Stopping Time") +
  xlab("Reported Assorter Mean") +
  scale_y_log10()


###### gaussian stopping times #######
full_data <- read_csv("all_gaussian_simulations.csv")
mean_data <- full_data %>%
  filter(sd == 0.05) %>%
  pivot_longer(cols = starts_with("mean_stop_"), names_prefix = "mean_stop_", names_to = "method", values_to = "stopping_time") %>%
  select(-...1) %>%
  filter(!(method %in% c("uinnsm_uniform", "uinnsm_vertex", "unstrat_fixed"))) %>%
  filter(!((method == "unstrat_agrapa") & (allocation_rule %in%  c("larger_means", "round_robin")))) %>%
  group_by(across(-c(rep, stopping_time))) %>%
  summarize(
    expected_stopping_time = mean(stopping_time), 
    prob_stop = mean(stopping_time < 1000),
    sd_stopping_time = sd(stopping_time), 
    min_stopping_time = min(stopping_time), 
    max_stopping_time = max(stopping_time)) %>%
  mutate(method = case_match(method, 
                             "lcb_fixed"~"LCB Fixed", 
                             "lcb_agrapa"~"LCB AGRAPA",
                             "uinnsm_adaptive"~"UI-NNSM Adaptive",
                             "uinnsm_fixed"~"UI-NNSM Fixed",
                             "unstrat_agrapa"~"Unstratified AGRAPA",
                             "unstrat_fixed"~"Unstratified Fixed")) %>%
  mutate(allocation_rule = case_match(allocation_rule, 
                                      "larger_means" ~ "Larger Means",
                                      "neyman" ~ "Neyman",
                                      "round_robin" ~ "Round Robin")) %>%
  mutate(delta = paste("Stratum gap =", delta)) %>%
  mutate(K = paste("K =", K)) %>%
  filter(allocation_rule == "Round Robin" | ((allocation_rule == "Neyman") & (method == "Unstratified AGRAPA"))) %>%
  rename(Method = method, `Allocation Rule` = allocation_rule)


expected_stop_plot <- ggplot(mean_data, aes(x = global_mean, y = expected_stopping_time, color = Method)) +
  geom_line(size = 1) +
  facet_grid(K ~ delta) +
  xlab("Global Mean") +
  ylab("Expected Stopping Time") +
  theme_bw() +
  theme(text = element_text(size = 18), axis.text = element_text(size = 14)) +
  scale_y_log10()


  



