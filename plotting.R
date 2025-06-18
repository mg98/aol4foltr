# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(tidyverse)
library(tikzDevice)
library(RColorBrewer)
library(lubridate)

print_plot <- function(plot, name, width=3.4, height=1.5){
  tex_name <- sprintf("results/fig/%s.tex", name)
  pdf_name <- sprintf("results/fig/%s.pdf", name)
  tex_width <- width; tex_height <- height
  pdf_width <- tex_width * 4; pdf_height <- tex_height * 4
  
  # Define LaTeX preamble to use sans-serif font
  sans_preamble <- c(
    "\\usepackage{pgfplots}",
    "\\pgfplotsset{compat=newest}",
    "\\usepackage[utf8]{inputenc}",
    "\\usepackage[T1]{fontenc}",
    "\\usepackage{sfmath}",  # Use sfmath for sans-serif math
    "\\renewcommand{\\familydefault}{\\sfdefault}"  # Set default font to sans-serif
  )
  
  # Use tikzDevice with custom preamble
  tikz(file = tex_name, width = tex_width, height = tex_height, sanitize = TRUE,
       documentDeclaration = "\\documentclass[12pt]{standalone}",
       packages = sans_preamble)
  print(plot)
  dev.off()
  
  # PDF export
  pdf(file = pdf_name, width = pdf_width, height = pdf_height)
  print(plot)
  dev.off()
}

# Read the data
results <- read_csv("results/mrr_results.csv")

# Reshape the data for plotting
results_long <- results %>%
  pivot_longer(
    cols = starts_with("mrr_"),
    names_to = "method",
    values_to = "mrr"
  ) %>%
  mutate(
    method = case_when(
      method == "mrr_real_async" ~ "Async.",
      method == "mrr_real_sync" ~ "Sync", 
      method == "mrr_iid_async" ~ "Async. (IID)",
      method == "mrr_iid_sync" ~ "Sync. (IID)"
    )
  )

# Define custom colors
method_colors <- c(
  "Async." = brewer.pal(4, "Set1")[2],
  "Sync" = brewer.pal(4, "Set1")[1],
  "Async. (IID)" = brewer.pal(4, "Set1")[4],
  "Sync. (IID)" = brewer.pal(4, "Set1")[3]
)

# Create MRR plot
mrr_plot <- ggplot(results_long, aes(x = event_index, y = mrr, linetype = method, color = method)) +
  geom_line(linewidth = 0.5) +
  scale_linetype_manual(values = c("solid", "dashed", "dotted", "dotdash")) +
  scale_color_manual(values = method_colors) +
  #xlim(0, 500) +
  labs(
    x = "Rounds",
    y = "Mean Reciprocal Rank",
    linetype = "Method",
    color = "Method"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    legend.margin = margin(t = -10),  # Reduce top margin of legend
    plot.margin = margin(b = 5)       # Reduce bottom margin of plot
  )

mrr_plot

print_plot(mrr_plot, "experiment", height=2.1)



aol_data <- read_csv("dataset/aol_dataset_top10000.csv", 
                     col_types = cols(
                       query_id = col_double(),
                       user_id = col_double(),
                       time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                       query = col_character(),
                       doc_id = col_character(),
                       candidate_doc_ids = col_character()
                     ))

# Filter for top users and create time bins
query_timeline <- aol_data %>%
  filter(user_id %in% top_users) %>%
  mutate(
    user_id = factor(user_id, levels = top_users),
    # Create daily bins for smoother visualization  
    time_bin = floor_date(time, "day")
  ) %>%
  group_by(user_id, time_bin) %>%
  summarise(
    query_count = n(),
    .groups = "drop"
  )

# Create the timeline plot with each user on a horizontal line
timeline_plot <- ggplot(query_timeline, aes(x = time_bin, y = user_id, size = query_count)) +
  geom_point(color = "#E41A1C", alpha = 0.7) +
  scale_size_area(max_size = 4, guide = "none") +
  labs(
    x = "Timestamp",
    y = "User ID",
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(linetype = "dashed", color = "grey80")
  )

timeline_plot
print_plot(timeline_plot, "query_bursts")





# Create cumulative user query count distribution plot
user_query_counts <- aol_data %>%
  group_by(user_id) %>%
  summarise(query_count = n()) %>%
  arrange(desc(query_count)) %>%
  mutate(users_with_at_least = row_number())

p <- ggplot(user_query_counts, aes(x = query_count, y = users_with_at_least)) +
  geom_line(color = "#377EB8", size = 1) +
  scale_x_log10(breaks = c(200, 500, 1000, 2000, 10000, 50000)) +
  scale_y_log10() +
  labs(
    x = "Number of clicks (x)",
    y = "Users with >=x clicks"
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank()
  )

print_plot(p, "user_queries")



# Read EMD data
data <- read_csv("results/feature_divergences.csv")

p <- data %>%
  ggplot(aes(x = feature_index + 1, y = divergence)) +
  geom_bar(stat = "identity", fill = "#377EB8", alpha = 0.7) +
  labs(
    x = "Feature index", 
    y = "Wasserstein distance"
  ) +
  theme_bw() +
  scale_x_continuous(breaks = seq(0, max(data$feature_index) + 1, by = 20))

print_plot(p, "divergence", height=1.5)

