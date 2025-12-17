# validation_against_StMoMo/stmomo_fit_cbd.R
# ------------------------------------------------------------
# Fit CBD (M5-ish) with StMoMo on France Excel dataset (3 sheets):
#   - "Exposure to risk"
#   - "Deaths"
# (Death Rate sheet not needed)
#
# Data file:
#   Data/data_france.xlsx
#
# Outputs:
#   validation_against_StMoMo/outputs/
#     cbd_kappas.csv
#     cbd_fitted_logitq.csv      (NOW: after m_to_q(q_to_m(.)) transform)
#     cbd_data_summary.json
# ------------------------------------------------------------

if ("package:demography" %in% search()) detach("package:demography", unload = TRUE)

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(tibble)
  library(StMoMo)
})

# -----------------------------
# Config
# -----------------------------
ROOT_DIR <- "."
DATA_DIR <- file.path(ROOT_DIR, "Data")
OUT_DIR  <- file.path(ROOT_DIR, "validation_against_StMoMo", "outputs")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

XLSX_PATH <- file.path(DATA_DIR, "data_france.xlsx")

SEX_COL <- "Total"  # "Total" / "Male" / "Female"

AGE_MIN <- 60
AGE_MAX <- 100
YEAR_MIN <- 1970
YEAR_MAX <- 2019

TARGET_AGES  <- seq(AGE_MIN, AGE_MAX, by = 1)
TARGET_YEARS <- seq(YEAR_MIN, YEAR_MAX, by = 1)

# -----------------------------
# Helpers
# -----------------------------
stop_if_missing_cols <- function(df, cols, sheet_name) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(sprintf("Sheet '%s' is missing columns: %s",
                 sheet_name, paste(missing, collapse = ", ")))
  }
}

clean_age <- function(x) {
  x <- as.character(x)
  x <- gsub("\\+", "", x)
  x <- gsub("[^0-9\\.]", "", x)
  suppressWarnings(as.numeric(x))
}

read_sheet_long <- function(path, sheet) {
  df <- read_excel(path, sheet = sheet, skip = 2)
  df <- as.data.frame(df)
  names(df) <- trimws(names(df))

  req <- c("Year", "Age", "Female", "Male", "Total")
  stop_if_missing_cols(df, req, sheet)

  df %>%
    mutate(
      Year = suppressWarnings(as.integer(Year)),
      Age  = clean_age(Age)
    ) %>%
    filter(!is.na(Year), !is.na(Age)) %>%
    select(Year, Age, Female, Male, Total)
}

apply_filters <- function(df) {
  df %>%
    filter(Age >= AGE_MIN, Age <= AGE_MAX) %>%
    filter(Year >= YEAR_MIN, Year <= YEAR_MAX)
}

to_matrix_age_year <- function(df, value_col, ages, years) {
  wide <- df %>%
    select(Year, Age, !!sym(value_col)) %>%
    pivot_wider(names_from = Year, values_from = !!sym(value_col)) %>%
    arrange(Age)

  for (yy in years) {
    yname <- as.character(yy)
    if (!(yname %in% names(wide))) wide[[yname]] <- NA_real_
  }

  wide <- wide %>% select(Age, all_of(as.character(years)))

  mat <- as.matrix(wide %>% select(-Age))
  storage.mode(mat) <- "double"
  rownames(mat) <- as.character(wide$Age)
  colnames(mat) <- as.character(years)
  mat
}

build_stmomo_data <- function(Dxt, Ext, ages, years, type = "central", series_label = "Total") {
  attempt <- tryCatch(
    StMoMo::StMoMoData(Dxt, Ext, ages, years, type = type),
    error = identity
  )
  if (!inherits(attempt, "error") && inherits(attempt, "StMoMoData")) return(attempt)

  manual <- list(
    Dxt = Dxt,
    Ext = Ext,
    ages = ages,
    years = years,
    ages.lab = as.character(ages),
    years.lab = as.character(years),
    type = type,
    series = series_label,
    label = "constructed_from_matrices"
  )
  class(manual) <- "StMoMoData"
  manual
}

# Python-consistent clip
clip01 <- function(x, lo = 1e-10, hi = 1 - 1e-10) pmin(pmax(x, lo), hi)

logit <- function(p) log(p / (1 - p))

# EXACT PYMORT conversions
m_to_q <- function(m) {
  q <- m / (1.0 + 0.5 * m)
  clip01(q)
}

q_to_m <- function(q) {
  q <- clip01(q)
  (2.0 * q) / (2.0 - q)
}

# -----------------------------
# Load data
# -----------------------------
if (!file.exists(XLSX_PATH)) stop(sprintf("Data file not found: %s", XLSX_PATH))

exposure_df <- read_sheet_long(XLSX_PATH, "Exposure to risk") %>% apply_filters()
deaths_df   <- read_sheet_long(XLSX_PATH, "Deaths")           %>% apply_filters()

joined <- exposure_df %>%
  rename(Ex_Female = Female, Ex_Male = Male, Ex_Total = Total) %>%
  inner_join(
    deaths_df %>% rename(Dx_Female = Female, Dx_Male = Male, Dx_Total = Total),
    by = c("Year", "Age")
  )

sex_map <- list(
  "Female" = list(Ex = "Ex_Female", Dx = "Dx_Female"),
  "Male"   = list(Ex = "Ex_Male",   Dx = "Dx_Male"),
  "Total"  = list(Ex = "Ex_Total",  Dx = "Dx_Total")
)
if (!(SEX_COL %in% names(sex_map))) stop("SEX_COL must be one of: 'Female', 'Male', 'Total'")
Ex_col <- sex_map[[SEX_COL]]$Ex
Dx_col <- sex_map[[SEX_COL]]$Dx

joined2 <- joined %>%
  transmute(
    Year = Year,
    Age  = Age,
    Ex   = suppressWarnings(as.numeric(.data[[Ex_col]])),
    Dx   = suppressWarnings(as.numeric(.data[[Dx_col]]))
  ) %>%
  filter(
    Age %in% TARGET_AGES,
    Year %in% TARGET_YEARS,
    is.finite(Year), is.finite(Age),
    is.finite(Ex), is.finite(Dx),
    Ex > 0,
    Dx >= 0
  )

ages  <- intersect(TARGET_AGES, sort(unique(joined2$Age)))
years <- intersect(TARGET_YEARS, sort(unique(joined2$Year)))
joined2 <- joined2 %>% filter(Age %in% ages, Year %in% years)

Ext <- to_matrix_age_year(joined2, "Ex", ages, years)
Dxt <- to_matrix_age_year(joined2, "Dx", ages, years)

good_age <- apply(Ext, 1, function(r) all(is.finite(r)) && all(r > 0)) &
            apply(Dxt, 1, function(r) all(is.finite(r)) && all(r >= 0))
Ext <- Ext[good_age, , drop = FALSE]
Dxt <- Dxt[good_age, , drop = FALSE]
ages_fit <- as.numeric(rownames(Ext))

good_year <- apply(Ext, 2, function(c) all(is.finite(c)) && all(c > 0)) &
             apply(Dxt, 2, function(c) all(is.finite(c)) && all(c >= 0))
Ext <- Ext[, good_year, drop = FALSE]
Dxt <- Dxt[, good_year, drop = FALSE]
years_fit <- as.integer(colnames(Ext))

if (length(ages_fit) < 5) stop("Too few ages left after filtering.")
if (length(years_fit) < 5) stop("Too few years left after filtering.")
if (anyNA(Ext) || anyNA(Dxt)) stop("Ext/Dxt contain NA after filtering")
if (any(Ext <= 0)) stop("Ext must be strictly positive")
if (any(Dxt < 0)) stop("Dxt must be non-negative")

# -----------------------------
# Fit CBD with StMoMo
# -----------------------------
StMoMo_dat <- build_stmomo_data(Dxt, Ext, ages_fit, years_fit, type = "central", series_label = SEX_COL)
stopifnot(inherits(StMoMo_dat, "StMoMoData"))

CBDmod <- StMoMo::cbd()
fit_cbd <- StMoMo::fit(CBDmod, data = StMoMo_dat, ages.fit = ages_fit, years.fit = years_fit)

kt <- fit_cbd$kt
if (is.vector(kt)) stop("Unexpected fit_cbd$kt is a vector. Check StMoMo version / model object.")
kappa1 <- as.numeric(kt[1, ]); names(kappa1) <- years_fit
kappa2 <- as.numeric(kt[2, ]); names(kappa2) <- years_fit

# Model systematic logit(q)
xbar <- mean(ages_fit)
age_centered <- ages_fit - xbar

logitq_hat <- matrix(NA_real_, nrow = length(ages_fit), ncol = length(years_fit))
for (t in seq_along(years_fit)) {
  logitq_hat[, t] <- kappa1[t] + kappa2[t] * age_centered
}

# Transform to be comparable with PYMORT:
# logit(q) -> q -> m -> q(m_to_q) -> logit(q)
q_hat  <- clip01(1.0 / (1.0 + exp(-logitq_hat)))
m_hat  <- q_to_m(q_hat)
q_hat2 <- m_to_q(m_hat)
logitq_hat2 <- logit(q_hat2)

rownames(logitq_hat2) <- as.character(ages_fit)
colnames(logitq_hat2) <- as.character(years_fit)

# -----------------------------
# Save outputs
# -----------------------------
kappas_df <- data.frame(Year = years_fit, kappa1 = kappa1, kappa2 = kappa2)
write.csv(kappas_df, file.path(OUT_DIR, "cbd_kappas.csv"), row.names = FALSE)

logitq_df <- as.data.frame(logitq_hat2) %>%
  rownames_to_column("Age") %>%
  mutate(Age = as.numeric(Age)) %>%
  pivot_longer(-Age, names_to = "Year", values_to = "logitq_fitted") %>%
  mutate(Year = as.integer(Year))
write.csv(logitq_df, file.path(OUT_DIR, "cbd_fitted_logitq.csv"), row.names = FALSE)

summary <- list(
  file = XLSX_PATH,
  sex = SEX_COL,
  ages_requested = list(min = AGE_MIN, max = AGE_MAX),
  ages_used = list(min = min(ages_fit), max = max(ages_fit), n = length(ages_fit)),
  years_used = list(min = min(years_fit), max = max(years_fit), n = length(years_fit)),
  xbar = xbar,
  notes = "StMoMo CBD fit using Dxt/Ext. Exported logit(q) after q->m->q transform to match PYMORT m_to_q definition."
)
writeLines(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE),
           file.path(OUT_DIR, "cbd_data_summary.json"))

message("Done. Outputs written to: ", OUT_DIR)
message(sprintf("Used ages %d..%d (n=%d), years %d..%d (n=%d)",
                min(ages_fit), max(ages_fit), length(ages_fit),
                min(years_fit), max(years_fit), length(years_fit)))