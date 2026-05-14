library(ggplot2)
library(dplyr)
library(xtable)

#data100 <- read.csv("results/factorial_results_100.csv")
#data500 <- read.csv("results/factorial_results_500.csv")
#data1000 <- read.csv("results/factorial_results_1000.csv")

data1_1000 <- read.csv("results/factorial_results_baseline_1000.csv")
data1_100 <- read.csv("results/factorial_results_baseline_100.csv")
data2_1000 <- read.csv("results/factorial_results_experienced_high_variability_1000.csv")
data2_100 <- read.csv("results/factorial_results_experienced_high_variability_100.csv")
data3_1000 <- read.csv("results/factorial_results_low_variability_more_aggressive_more_conservative_1000.csv")
data3_100 <- read.csv("results/factorial_results_low_variability_more_aggressive_more_conservative_100.csv")
data4_1000 <- read.csv("results/factorial_results_urban_heavy_rural_heavy_1000.csv")
data4_100 <- read.csv("results/factorial_results_urban_heavy_rural_heavy_100.csv")
data5_1000 <- read.csv("results/factorial_results_high_caseload_1000.csv")
data5_100 <- read.csv("results/factorial_results_high_caseload_100.csv")

data <- rbind(data1_1000,data2_1000,data3_1000,data4_1000,data5_1000,
              data1_100,data2_100,data3_100,data4_100,data5_100)

g1 <- ggplot(data=data,aes(x=response_rate,y=coverage_95)) + facet_grid(population_scenario ~ n_derms) + geom_line(aes(color=mechanism)) + geom_hline(yintercept = 95) + theme_bw() + scale_x_reverse(limits=c(0.9,0.1),breaks=(seq(10,1,-1)/10))

ggsave("plots/coverage_pct.png",g1)

data$sig <- ifelse(data$dunnett_lower > 0 | data$dunnett_upper < 0, "significant","not")

g2 <- ggplot(data=data[data$n_derms == 100, ],aes(x=mean_est,y=response_rate*100,xmin=dunnett_lower,xmax=dunnett_upper)) + geom_point(size=0.5) + 
    geom_errorbar(aes(color=sig),width=3) + 
    facet_grid(population_scenario ~ mechanism,labeller = as_labeller(c(baseline="baseline",experienced="experienced",  MCAR="MCAR",MNAR="MNAR",high_caseload="high\ncaseload", high_variability="high\nvariability", low_variability="low\nvariability", more_aggressive="more\naggressive", more_conservative="more\nconservative", urban_heavy="urban\nheavy", rural_heavy="rural\n heavy"))) + 
    theme_bw() + coord_flip() + theme(legend.position = "none") +
    scale_color_manual(values=c("significant" = "#CC79A7", "not significant" = "#0072B2")) +
    scale_y_reverse(limits=c(92.5,27.5),breaks=(seq(100,21,-10))) + 
    geom_vline(xintercept=0,alpha=0.5,linewidth=0.35,linetype=2,color="darkgreen") + 
    labs(title="Dunnett 95% confidence intervals (bias)", subtitle="100 dermatologists", x="Bias", y="Response Rate") #+ scale_x_reverse(limits=c(0.9,0.1),breaks=(seq(10,1,-1)/10))
ggsave("plots/CIs_100derms.png",g2,width=12,height=10)

g3 <- ggplot(data=data[data$n_derms == 1000, ],aes(x=mean_est,y=response_rate*100,xmin=dunnett_lower,xmax=dunnett_upper)) + geom_point(size=0.5) + 
    geom_errorbar(aes(color=sig),width=3) + 
    facet_grid(population_scenario ~ mechanism,labeller = as_labeller(c(baseline="baseline",experienced="experienced",  MCAR="MCAR",MNAR="MNAR",high_caseload="high\ncaseload", high_variability="high\nvariability", low_variability="low\nvariability", more_aggressive="more\naggressive", more_conservative="more\nconservative", urban_heavy="urban\nheavy", rural_heavy="rural\n heavy"))) + 
    theme_bw() + coord_flip() + theme(legend.position = "none") +
    scale_color_manual(values=c("significant" = "#CC79A7", "not significant" = "#0072B2")) +
    scale_y_reverse(limits=c(92.5,7.5),breaks=(seq(100,1,-10))) + 
    geom_vline(xintercept=0,alpha=0.5,linewidth=0.35,linetype=2,color="darkgreen") + 
    labs(title="Dunnett 95% confidence intervals (bias)", subtitle="1000 dermatologists", x="Bias", y="Response Rate") #+ scale_x_reverse(limits=c(0.9,0.1),breaks=(seq(10,1,-1)/10))
ggsave("plots/CIs_1000derms.png",g3,width=12,height=10)

data$sd_from_true <- abs(data$mean_est)/sqrt(data$beta_var)

for(rate in 3:9)
{
    for(mech in c("MCAR","MNAR"))
    {
        xt <- xtable(data[data$mechanism == mech & data$response_rate == rate/10 & data$n_derm == 1000,c("population_scenario","rel_bias_pct","response_rate","dunnett_lower","dunnett_upper","sd_from_true")])
        digits(xt) <- c(0,0,3,0,3,3,2)
        xt$response_rate <- xt$response_rate*100
        print(xt, type = "latex", file = paste0("results/", mech, "_1000_", as.character(rate), ".tex"), include.rownames=FALSE)
    }
}

for(rate in 1:9)
{
    for(mech in c("MCAR","MNAR"))
    {
        xt <- xtable(data[data$mechanism == mech & data$response_rate == rate/10 & data$n_derm == 1000,c("population_scenario","rel_bias_pct","response_rate","dunnett_lower","dunnett_upper","sd_from_true")])
digits(xt) <- c(0,0,3,0,3,3,2)
        xt$response_rate <- xt$response_rate*100
        print(xt, type = "latex", file = paste0("results/", mech, "_1000_", as.character(rate), ".tex"), include.rownames=FALSE)
    }
}


