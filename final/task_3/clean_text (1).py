import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Κατέβασμα της λίστας των αγγλικών stopwords (εκτελείται αυτόματα την πρώτη φορά)
nltk.download('stopwords')
# Μετατροπή της λίστας σε σύνολο (set) για πολύ πιο γρήγορη αναζήτηση κατά την εκτέλεση
stop_words = set(stopwords.words('english'))

def clean_comment(text):
    """
    Αυτή η συνάρτηση δέχεται ένα ακατέργαστο κείμενο και εφαρμόζει
    διαδοχικά φίλτρα καθαρισμού για την προετοιμασία του μοντέλου.
    """
    # Έλεγχος ασφαλείας: Αν το κείμενο είναι κενό ή NaN, επιστρέφουμε μια κενή συμβολοσειρά
    if not isinstance(text, str):
        return ""
    
    # 1. Μετατροπή σε πεζά (Lowercase): Μετατρέπουμε όλο το κείμενο σε μικρά γράμματα
    text = text.lower()
    
    # 2. Αφαίρεση URLs: Διαγράφουμε οποιονδήποτε διαδικτυακό σύνδεσμο
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Αφαίρεση Mentions: Διαγράφουμε τις αναφορές σε χρήστες (π.χ. @username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Αφαίρεση Hashtags: Διαγράφουμε λέξεις που ξεκινούν με το σύμβολο #
    text = re.sub(r'#\w+', '', text)
    
    # 5. Αφαίρεση Ειδικών Χαρακτήρων: Κρατάμε αποκλειστικά γράμματα και κενά διαστήματα
    text = re.sub(r'[^\w\s]', '', text)
    
    # 6. Αφαίρεση Stopwords: Αφαιρούμε τις πολύ κοινές λέξεις (the, is, in, at κ.λπ.)
    # Χωρίζουμε το κείμενο σε μεμονωμένες λέξεις, φιλτράρουμε και το ξαναενώνουμε
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    text = ' '.join(cleaned_words)
    
    return text

def main():
    # Φόρτωση του αρχείου με τα δεδομένα (μπορείς να βάλεις το αρχείο με τα annotations σου)
    input_file = "Final_Concatenated_File_Cleaned.csv"
    print(f"📥 Γίνεται ανάγνωση των δεδομένων από το αρχείο: {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Το αρχείο '{input_file}' δεν βρέθηκε. Βεβαιώσου ότι το όνομα είναι το σωστό.")
        return

    print("🧹 Ξεκινάει ο βαθύς καθαρισμός του κειμένου. Η διαδικασία μπορεί να πάρει λίγα δευτερόλεπτα...")
    
    # Εφαρμογή της συνάρτησης καθαρισμού δημιουργώντας μια νέα στήλη 'cleaned_text'
    # Με αυτόν τον τρόπο διατηρούμε ανέπαφο και το αρχικό κείμενο για σύγκριση
    df['cleaned_text'] = df['text'].apply(clean_comment)

    # Αποθήκευση των πλήρως καθαρισμένων δεδομένων σε ένα νέο τελικό αρχείο CSV
    output_file = "eurovision_FINAL_cleaned_data.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print("=" * 80)
    print("✅ Το Βήμα 3 (Text Cleaning) ολοκληρώθηκε με απόλυτη επιτυχία!")
    print(f"📁 Το νέο αρχείο αποθηκεύτηκε με το όνομα: {output_file}")
    print("=" * 80)
    
    # Εμφάνιση ενός μικρού δείγματος στο τερματικό για άμεση οπτική επαλήθευση του αποτελέσματος
    print("\n🔍 Αναλυτικό Δείγμα Καθαρισμού (Πριν και Μετά):")
    for idx, row in df.head(5).iterrows():
        print(f"\n--- Σχόλιο {idx + 1} ---")
        print(f"ΑΡΧΙΚΟ : {row['text']}")
        print(f"ΚΑΘΑΡΟ : {row['cleaned_text']}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()