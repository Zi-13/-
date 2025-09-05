# Re-run the plotting code from cell aeec67e0 to display with Chinese characters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for boxplots

# Ensure df_male_fetus is available and has '孕妇代码', '检测孕周_数值', 'Y染色体浓度', '孕妇BMI'
# If using sampled data, replace df_male_fetus with df_sample
# Assuming df_male_fetus is the primary data source for this analysis

# Ensure convert_gestational_age is defined or available
try:
    convert_gestational_age
except NameError:
    def convert_gestational_age(ga_str):
        """将'13w+5'格式转换为13.71周"""
        try:
            if isinstance(ga_str, (int, float)):
                return float(ga_str)
            elif 'w+' in str(ga_str):
                weeks, days = map(int, str(ga_str).split('w+'))
                return weeks + days/7
            elif 'w' in str(ga_str):
                return float(str(ga_str).replace('w', ''))
            else:
                return float(ga_str)
        except:
            return np.nan  # 处理异常情况

# Ensure '检测孕周_数值' is available in df_male_fetus
if '检测孕周_数值' not in df_male_fetus.columns:
     print("Adding '检测孕周_数值' column to df_male_fetus...")
     df_male_fetus['检测孕周_数值'] = df_male_fetus['检测孕周'].apply(convert_gestational_age)

# Ensure equal_width_binning and decision_tree_binning functions are defined or available
try:
    equal_width_binning
    calculate_time_to_target
    decision_tree_binning
except NameError:
    print("Binning functions not found. Please run the cells defining equal_width_binning, calculate_time_to_target, and decision_tree_binning.")
    # Stop execution or inform user if functions are not available
    raise # Raise error to stop if functions are missing


# --- Apply Equal Width Binning ---
n_equal_bins = 4 # Number of bins for equal width
equal_bins_edges = equal_width_binning(df_male_fetus['孕妇BMI'], n_bins=n_equal_bins)
print(f"\nEqual Width Bin Edges ({n_equal_bins} bins):")
print(equal_bins_edges)

# Apply bins to the DataFrame
df_male_fetus['BMI_Group_Equal'] = pd.cut(
    df_male_fetus['孕妇BMI'],
    bins=equal_bins_edges,
    include_lowest=True,
    labels=[f'Equal_Bin_{i+1}' for i in range(len(equal_bins_edges)-1)]
)

print("\nDataFrame head with Equal Width BMI Groups:")
display(df_male_fetus[['孕妇BMI', 'BMI_Group_Equal']].head())


# --- Apply Decision Tree Binning ---
n_dt_bins = 4 # Number of bins for decision tree (results in n_bins-1 splits)

# Calculate time to target Y concentration for decision tree binning
print("\nCalculating time to target Y concentration for Decision Tree Binning...")
target_time_df = calculate_time_to_target(df_male_fetus)
print("Time to target DataFrame head (for Decision Tree Binning):")
display(target_time_df.head())

# Use decision tree binning to get optimal splits
if not target_time_df.empty:
    optimal_bmi_splits = decision_tree_binning(
        target_time_df['孕妇BMI'],
        target_time_df['达标时间'],
        n_bins=n_dt_bins
    )
    print(f"\nOptimal BMI Split Points for Decision Tree Binning ({n_dt_bins-1} splits):")
    print(optimal_bmi_splits)

    # Create bin edges including min and max BMI for pd.cut
    min_bmi = df_male_fetus['孕妇BMI'].min()
    max_bmi = df_male_fetus['孕妇BMI'].max()
    dt_bins_edges = [min_bmi - 1e-6] + optimal_bmi_splits + [max_bmi + 1e-6] # Add boundaries slightly outside min/max

    # Apply bins to the DataFrame (merge target_time_df back or apply to original df if BMI is consistent)
    # Applying to original df_male_fetus assuming BMI doesn't change significantly over time for a patient
    df_male_fetus['BMI_Group_DT'] = pd.cut(
        df_male_fetus['孕妇BMI'],
        bins=dt_bins_edges,
        include_lowest=True,
        labels=[f'DT_Bin_{i+1}' for i in range(len(dt_bins_edges)-1)]
    )

    print("\nDataFrame head with Decision Tree BMI Groups:")
    display(df_male_fetus[['孕妇BMI', 'BMI_Group_DT']].head())

else:
    print("\nCould not calculate time to target, skipping Decision Tree Binning application.")
    df_male_fetus['BMI_Group_DT'] = pd.NA # Assign Not Available if DT binning not applied


# --- Visualize Binning Effects ---

# Scatter plot of Y_concentration vs 检测孕周_数值, colored by BMI Group (Equal Width)
plt.figure(figsize=(14, 6))
sns.scatterplot(data=df_male_fetus, x='检测孕周_数值', y='Y染色体浓度', hue='BMI_Group_Equal', alpha=0.6, palette='viridis')
plt.title('Y染色体浓度 vs 检测孕周 (按等宽 BMI 分组)')
plt.xlabel('检测孕周 (数值)')
plt.ylabel('Y染色体浓度')
plt.grid(True, alpha=0.3)
# Add vertical lines for Equal Width BMI bin edges (optional, can clutter the plot)
# for edge in equal_bins_edges[1:-1]:
#     plt.axvline(x=df_male_fetus['检测孕周_数值'].max(), ymin=0, ymax=1, color='gray', linestyle='--', alpha=0.5, label=f'BMI={edge:.2f}')
plt.show()

# Scatter plot of Y_concentration vs 检测孕周_数值, colored by BMI Group (Decision Tree)
if 'BMI_Group_DT' in df_male_fetus.columns and not df_male_fetus['BMI_Group_DT'].isnull().all():
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=df_male_fetus, x='检测孕周_数值', y='Y染色体浓度', hue='BMI_Group_DT', alpha=0.6, palette='viridis')
    plt.title('Y染色体浓度 vs 检测孕周 (按决策树 BMI 分组)')
    plt.xlabel('检测孕周 (数值)')
    plt.ylabel('Y染色体浓度')
    plt.grid(True, alpha=0.3)
    # Add vertical lines for Decision Tree BMI split points (optional)
    # for split in optimal_bmi_splits:
    #      plt.axvline(x=df_male_fetus['检测孕周_数值'].max(), ymin=0, ymax=1, color='gray', linestyle='--', alpha=0.5, label=f'BMI split={split:.2f}')
    plt.show()


# Box plot of Y_concentration by BMI Group (Equal Width)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_male_fetus, x='BMI_Group_Equal', y='Y染色体浓度', palette='viridis')
plt.title('Y染色体浓度 分布 (按等宽 BMI 分组)')
plt.xlabel('等宽 BMI 分组')
plt.ylabel('Y染色体浓度')
plt.grid(True, alpha=0.3)
plt.show()

# Box plot of Y_concentration by BMI Group (Decision Tree)
if 'BMI_Group_DT' in df_male_fetus.columns and not df_male_fetus['BMI_Group_DT'].isnull().all():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_male_fetus, x='BMI_Group_DT', y='Y染色体浓度', palette='viridis')
    plt.title('Y染色体浓度 分布 (按决策树 BMI 分组)')
    plt.xlabel('决策树 BMI 分组')
    plt.ylabel('Y染色体浓度')
    plt.grid(True, alpha=0.3)
    plt.show()


# --- Summarize Features by Group ---

print("\nSummary Statistics by Equal Width BMI Group:")
if 'BMI_Group_Equal' in df_male_fetus.columns:
     equal_group_stats = df_male_fetus.groupby('BMI_Group_Equal')[['孕妇BMI', '检测孕周_数值', 'Y染色体浓度']].agg(['mean', 'std', 'count'])
     display(equal_group_stats)
else:
     print("Equal Width BMI groups not available.")


print("\nSummary Statistics by Decision Tree BMI Group:")
if 'BMI_Group_DT' in df_male_fetus.columns and not df_male_fetus['BMI_Group_DT'].isnull().all():
    # Need to calculate '达标时间' per patient and merge with BMI groups for this summary
    if '达标时间' not in target_time_df.columns: # Recalculate if necessary
         print("Calculating time to target for summary...")
         target_time_df = calculate_time_to_target(df_male_fetus)

    # Merge target_time_df with the df_male_fetus to get BMI groups for each patient's target time
    # Ensure both dataframes have '孕妇代码' and merge on it
    target_time_with_groups = target_time_df.merge(
        df_male_fetus[['孕妇代码', 'BMI_Group_DT']].drop_duplicates(subset=['孕妇代码']),
        on='孕妇代码',
        how='left'
    )
    # Filter out patients for whom target time was not calculated or DT group was not assigned
    target_time_with_groups = target_time_with_groups.dropna(subset=['达标时间', 'BMI_Group_DT'])

    if not target_time_with_groups.empty:
         dt_group_stats = target_time_with_groups.groupby('BMI_Group_DT')[['孕妇BMI', '达标时间']].agg(['mean', 'std', 'count'])
         display(dt_group_stats)
    else:
         print("No valid data for Decision Tree BMI group summary.")

else:
     print("Decision Tree BMI groups or target time data not available for summary.")