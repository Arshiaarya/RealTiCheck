import sys
import csv
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

def get_threshold_prediction(votes, threshold, default="Not Supported"):
    """Get prediction based on threshold of Not Supported votes."""
    if not votes:
        return default
    total_votes = len(votes)
    not_supported_count = sum(1 for vote in votes if vote == 'Not Supported')
    not_supported_percentage = (not_supported_count / total_votes) * 100
    return 'Not Supported' if not_supported_percentage >= threshold else 'Supported'

def load_merged_data(file_path):
    """Load data from the CSV file with specified columns."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    data = []
    unique_claims = set()
    total_rows = 0
    
    label_mapping = {
        'supported': 'Supported',
        'refuted': 'Not Supported',
        'conflicting evidence/cherrypicking': 'Not Supported',
        'not enough evidence': 'Not Supported',
        'Supported': 'Supported',
        'Not Supported': 'Not Supported',
        'not supported':'Not Supported',
        'error': 'Not Supported'  # Map 'error' to 'Not Supported'
    }
    
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1
            
            # Normalize claim and get label
            claim = row.get('claim', '').strip().lower()
            original_label = row.get('matched_label', '').strip().lower()
            
            # Skip rows without claims or labels
            if not claim or not original_label:
                continue
                
            # Map the label
            mapped_label = label_mapping.get(original_label)
            if mapped_label is None:
                print(f"Warning: Unknown label found: {original_label}")
                continue
            
            # Get entailment and normalize it
            entailment = row.get('Entailment', '').strip()
            if entailment.lower() in ['false', 'not supported']:
                mapped_entailment = 'Not Supported'
            elif entailment.lower() in ['true', 'supported']:
                mapped_entailment = 'Supported'
            else:
                # Skip rows with invalid entailment
                if not entailment:
                    continue
                mapped_entailment = entailment
            
            # Get article text (instead of paragraph)
            article_text = row.get('article_text', '').strip()
            if not article_text:
                continue
                
            relevance = row.get('Relevance', 'Unknown').strip().lower()
            if relevance == 'irrelevant':
                continue
                
            data.append({
                'claim': claim,
                'label': mapped_label,
                'entailment': mapped_entailment,
                'article_text': article_text,
                'query': row.get('query', '').strip(),
                'justification': row.get('Justification', '').strip()
            })
            
            unique_claims.add(claim)
    
    print(f"Loaded {len(data)} rows")
    print(f"Found {len(unique_claims)} unique claims")
    
    return data

def calculate_metrics_with_thresholding(data, claim_threshold, verbose=True):
    """
    Calculate metrics using only thresholding (no majority voting).
    """
    # Group all entailments by claim
    claim_entailments = defaultdict(list)
    claim_labels = {}
    
    for row in data:
        claim = row['claim']
        entailment = row['entailment']
        label = row['label']
        
        claim_entailments[claim].append(entailment)
        claim_labels[claim] = label
    
    # Apply thresholding at claim level
    claim_verdicts = {}
    if verbose:
        print(f"\nResults with threshold of {claim_threshold}%:")
    
    for claim, entailments in claim_entailments.items():
        claim_verdicts[claim] = get_threshold_prediction(entailments, claim_threshold)
        
        if verbose:
            total_votes = len(entailments)
            not_supported_count = sum(1 for vote in entailments if vote == 'Not Supported')
            not_supported_percentage = (not_supported_count / total_votes) * 100
            
            print(f"\nClaim: {claim}")
            print(f"All votes: {Counter(entailments)}")
            print(f"Not Supported percentage: {not_supported_percentage:.1f}%")
            print(f"Threshold: {claim_threshold}%")
            print(f"Final verdict: {claim_verdicts[claim]}")
    
    # Calculate metrics
    y_true = []
    y_pred = []
    for claim, verdict in claim_verdicts.items():
        y_true.append(claim_labels[claim])
        y_pred.append(verdict)
    
    # Calculate weighted metrics
    weighted_precision = precision_score(y_true, y_pred, pos_label='Not Supported', average='weighted')
    weighted_recall = recall_score(y_true, y_pred, pos_label='Not Supported', average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, pos_label='Not Supported', average='weighted')
    
    # Calculate binary metrics (Not Supported as positive class)
    binary_precision = precision_score(y_true, y_pred, pos_label='Not Supported')
    binary_recall = recall_score(y_true, y_pred, pos_label='Not Supported')
    binary_f1 = f1_score(y_true, y_pred, pos_label='Not Supported')
    
    # Calculate micro and macro metrics
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Create confusion matrix
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 'Not Supported' and y_pred[i] == 'Not Supported')
    fp = sum(1 for i in range(len(y_true)) if y_true[i] != 'Not Supported' and y_pred[i] == 'Not Supported')
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 'Not Supported' and y_pred[i] != 'Not Supported')
    tn = sum(1 for i in range(len(y_true)) if y_true[i] != 'Not Supported' and y_pred[i] != 'Not Supported')
    
    return {
        'metrics': {
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            },
            'binary': {
                'precision': binary_precision,
                'recall': binary_recall,
                'f1': binary_f1
            },
            'micro': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            }
        },
        'predictions': Counter(y_pred),
        'true_labels': Counter(y_true),
        'total_claims': len(y_pred),
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        },
        'raw_data': {
            'y_true': y_true,
            'y_pred': y_pred
        }
    }

def find_optimal_threshold(data, output_prefix="", output_dir="./results"):
    """Find the optimal threshold by testing a range of values and collect all metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = np.arange(0.1, 100.1, 1)  # Test thresholds from 0 to 100
    #thresholds = [51.0]
    threshold_results = []
    
    # Store F1 scores for each metric type
    binary_f1_scores = []
    micro_f1_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []
    
    print(f"\nTesting different thresholds for {output_prefix}...")
    for threshold in thresholds:
        # Use verbose=False to avoid cluttering the output
        results = calculate_metrics_with_thresholding(data, threshold, verbose=False)
        
        # Collect all F1 scores
        binary_f1_scores.append(results['metrics']['binary']['f1'])
        micro_f1_scores.append(results['metrics']['micro']['f1'])
        macro_f1_scores.append(results['metrics']['macro']['f1'])
        weighted_f1_scores.append(results['metrics']['weighted']['f1'])
        
        threshold_results.append((threshold, results['metrics']['macro']['f1'], results))
        
        if threshold % 10 == 0:
            print(f"Threshold {threshold}%: Macro F1 = {results['metrics']['macro']['f1']:.3f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    plt.rcParams.update({'font.size': 20})

    binary_optimal_idx = np.argmax(binary_f1_scores)
    micro_optimal_idx = np.argmax(micro_f1_scores)
    macro_optimal_idx = np.argmax(macro_f1_scores)
    weighted_optimal_idx = np.argmax(weighted_f1_scores)

    plt.plot(thresholds, binary_f1_scores, 'y-', label='Binary F1 (Optimal Threshold: {:.1f}%)'.format(thresholds[binary_optimal_idx]), linewidth=3)
    plt.plot(thresholds, micro_f1_scores, 'b-', label='Micro F1 (Optimal Threshold: {:.1f}%)'.format(thresholds[micro_optimal_idx]), linewidth=3)
    plt.plot(thresholds, macro_f1_scores, 'g-', label='Macro F1 (Optimal Threshold: {:.1f}%)'.format(thresholds[macro_optimal_idx]), linewidth=3)
    plt.plot(thresholds, weighted_f1_scores, 'r-', label='Weighted F1 (Optimal Threshold: {:.1f}%)'.format(thresholds[weighted_optimal_idx]), linewidth=3)

    # Find and plot optimal thresholds for each metric)

    # Remove labels from axvline by setting label=None
    plt.axvline(x=thresholds[binary_optimal_idx], color='y', linestyle='--')
    plt.axvline(x=thresholds[micro_optimal_idx], color='b', linestyle='--')
    plt.axvline(x=thresholds[macro_optimal_idx], color='g', linestyle='--')
    plt.axvline(x=thresholds[weighted_optimal_idx], color='r', linestyle='--')

    plt.xlabel('Percentage of "Not Supported" Votes Required', fontsize=24)
    plt.ylabel('F1 Score', fontsize=24)
   #plt.title(f'F1 Scores Across Different Averaging Methods {output_prefix}', fontsize=18)
    plt.grid(True)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(loc='lower right', fontsize=16)
    
    
    plt.tight_layout()
    
    # Save with appropriate filename based on prefix
    filename_prefix = output_prefix.replace(" ", "_").lower() if output_prefix else ""
    if filename_prefix:
        filename_prefix += "_"
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}f1_scores_comparison.png'), bbox_inches='tight', dpi=300)
    
    # Save the detailed metrics data
    results_df = pd.DataFrame({
        'Threshold': thresholds,
        'Binary_F1': binary_f1_scores,
        'Micro_F1': micro_f1_scores,
        'Macro_F1': macro_f1_scores,
        'Weighted_F1': weighted_f1_scores
    })
    results_df.to_csv(os.path.join(output_dir, f'{filename_prefix}threshold_detailed_results.csv'), index=False)
    
    # Find optimal threshold based on weighted F1
    optimal_threshold, best_weighted_f1, best_results = max(threshold_results, key=lambda x: x[1])
    return optimal_threshold, best_weighted_f1, best_results

def plot_confusion_matrix(results, title, output_dir="./results"):
    """Create a visualization of the confusion matrix."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cm = results['confusion_matrix']
    
    # Create confusion matrix array
    cm_array = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im)
    
    plt.xticks([0, 1], ['Supported', 'Not Supported'])
    plt.yticks([0, 1], ['Supported', 'Not Supported'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm_array.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_array[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_array[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save the confusion matrix
    filename = title.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, f'{filename}_confusion_matrix.png'), bbox_inches='tight', dpi=300)

def export_metrics_to_csv(all_results, filename, output_dir="./results"):
    """Export all metrics to a CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    
    for scenario_name, results in all_results.items():
        # Extract metrics and confusion matrix
        metrics = results['metrics']
        cm = results['confusion_matrix']
        
        # Create a row for this scenario
        row = {
            'Scenario': scenario_name,
            # Weighted metrics
            'Weighted_Precision': metrics['weighted']['precision'],
            'Weighted_Recall': metrics['weighted']['recall'],
            'Weighted_F1': metrics['weighted']['f1'],
            # Binary metrics
            'Binary_Precision': metrics['binary']['precision'],
            'Binary_Recall': metrics['binary']['recall'],
            'Binary_F1': metrics['binary']['f1'],
            # Micro metrics
            'Micro_Precision': metrics['micro']['precision'],
            'Micro_Recall': metrics['micro']['recall'],
            'Micro_F1': metrics['micro']['f1'],
            # Macro metrics
            'Macro_Precision': metrics['macro']['precision'],
            'Macro_Recall': metrics['macro']['recall'],
            'Macro_F1': metrics['macro']['f1'],
            # Confusion matrix
            'True_Positives': cm['true_positives'],
            'False_Positives': cm['false_positives'],
            'False_Negatives': cm['false_negatives'],
            'True_Negatives': cm['true_negatives'],
            # Additional info
            'Total_Claims': results['total_claims'],
            'Not_Supported_Predictions': results['predictions'].get('Not Supported', 0),
            'Supported_Predictions': results['predictions'].get('Supported', 0),
            'Not_Supported_True': results['true_labels'].get('Not Supported', 0),
            'Supported_True': results['true_labels'].get('Supported', 0)
        }
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"All metrics saved to {os.path.join(output_dir, filename)}")

def main():
    # Set file paths and output directory
    input_file = 'articles_merged_with_entailment_with_relevance_without_news_filtering_with_labels_deduplicated_fixed.csv'
    output_dir = './article_level_results'
    
    # Dictionary to store all results for comparison
    all_results = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n=== PROCESSING DATA ===")
    print("======================")
    data = load_merged_data(input_file)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold, best_f1, optimal_results = find_optimal_threshold(
        data, "Thresholding Analysis", output_dir
    )
    all_results["Optimal Threshold"] = optimal_results
    
    print(f"\nOptimal threshold: {optimal_threshold:.1f}%")
    print(f"Best weighted F1 score: {best_f1:.3f}")
    
    # Print optimal threshold results
    optimal_weighted = optimal_results['metrics']['macro']
    optimal_binary = optimal_results['metrics']['binary']
    
    print("\nMacro Metrics at Optimal Threshold:")
    print(f"Precision: {optimal_weighted['precision']:.3f}")
    print(f"Recall: {optimal_weighted['recall']:.3f}")
    print(f"F1 Score: {optimal_weighted['f1']:.3f}")
    print(optimal_results['confusion_matrix'])

    print("\nBinary Metrics at Optimal Threshold (Not Supported as positive class):")
    print(f"Precision: {optimal_binary['precision']:.3f}")
    print(f"Recall: {optimal_binary['recall']:.3f}")
    print(f"F1 Score: {optimal_binary['f1']:.3f}")
    
    # Plot confusion matrix for optimal threshold
    plot_confusion_matrix(optimal_results, f"Optimal Threshold {optimal_threshold:.1f}%", output_dir)
    
    # Try some specific thresholds of interest
    thresholds_of_interest = [0, 25, 50, 75, 100]
    for threshold in thresholds_of_interest:
        print(f"\nCalculating metrics for threshold {threshold}%...")
        results = calculate_metrics_with_thresholding(data, threshold, verbose=False)
        all_results[f"Threshold {threshold}%"] = results
        
        # Plot confusion matrix for this threshold
        plot_confusion_matrix(results, f"Threshold {threshold}%", output_dir)
    
    # Export metrics to CSV for comparison
    print("\nExporting metrics to CSV...")
    export_metrics_to_csv(all_results, "threshold_results_comparison.csv", output_dir)
    
    # Print summary
    print("\n=== SUMMARY OF RESULTS ===")
    print("=========================")
    for scenario_name, results in all_results.items():
        weighted_f1 = results['metrics']['weighted']['f1']
        binary_f1 = results['metrics']['binary']['f1']
        cm = results['confusion_matrix']
        
        print(f"{scenario_name}:")
        print(f"  Weighted F1: {weighted_f1:.3f}")
        print(f"  Binary F1 (Not Supported as positive): {binary_f1:.3f}")
        
        # Print confusion matrix
        print(f"  Confusion Matrix:")
        print(f"    True Positives (TP): {cm['true_positives']}")
        print(f"    False Positives (FP): {cm['false_positives']}")
        print(f"    False Negatives (FN): {cm['false_negatives']}")
        print(f"    True Negatives (TN): {cm['true_negatives']}")
        print(f"    Accuracy: {(cm['true_positives'] + cm['true_negatives']) / results['total_claims']:.3f}")
        
    print(f"\nAll results saved to {output_dir}/")

if __name__ == "__main__":
    main()
