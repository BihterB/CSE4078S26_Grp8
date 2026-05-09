import pandas as pd 
import os 
# This helper script reads baseline_summary.csv 
# and creates a cleaner table for presentation. 
INPUT_FILE = "results/baseline_summary.csv" 
OUTPUT_FILE = "results/presentation_result_table.csv" 
def main(): 
if not os.path.exists(INPUT_FILE): 
print("Summary file not found:", INPUT_FILE) 
        return 
 
    df = pd.read_csv(INPUT_FILE) 
 
    table = df[ 
        [ 
            "model_name", 
            "examples", 
            "avg_rouge_l", 
            "avg_tfidf_similarity", 
            "error_count", 
        ] 
    ].copy() 
 
    table = table.rename( 
        columns={ 
            "model_name": "Model", 
            "examples": "Examples", 
            "avg_rouge_l": "ROUGE-L", 
            "avg_tfidf_similarity": "TF-IDF Similarity", 
            "error_count": "Errors", 
        } 
    ) 
 
    table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8") 
 
    print("Presentation table:") 
    print(table) 
 
    print("\nSaved file:") 
    print(OUTPUT_FILE) 
 
 
if __name__ == "__main__": 
    main() 