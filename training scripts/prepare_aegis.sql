-- =====================================================================
-- prepare_aegis.sql
-- Prepares NVIDIA Aegis 2.0 for multi-label risk-category classification.
-- Dialect: DuckDB (runs from train_aegis.py, or the DuckDB CLI).
--
-- INPUT tables (one row per prompt/response pair, Aegis's original columns):
--   raw_train, raw_validation, raw_test
--
-- OUTPUT tables:
--   categories        final category list with training counts
--   train_prep        id, prompt, cats (list of categories)
--   validation_prep   same columns
--   test_prep         same columns
--
-- Standalone use in the DuckDB CLI: create the raw tables first, e.g.
--   CREATE TABLE raw_train AS SELECT * FROM read_json_auto('train.json');
--   (same for validation.json and test.json), then: .read prepare_aegis.sql
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. Settings
--    Categories with fewer training prompts than this are dropped:
--    too few examples to learn from or to measure reliably.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE settings AS SELECT 50 AS min_count;


-- ---------------------------------------------------------------------
-- 2. Helper: turn the comma-separated string into a clean list.
--    "Violence, Needs Caution, Harassment" -> ['Violence', 'Harassment']
--    Removes blanks and the two vague labels nobody can act on.
-- ---------------------------------------------------------------------
CREATE OR REPLACE MACRO clean_cats(s) AS
    list_filter(
        list_transform(string_split(coalesce(s, ''), ','), c -> trim(c)),
        c -> c NOT IN ('', 'Needs Caution', 'Other')
    );


-- ---------------------------------------------------------------------
-- 3. Stack the three splits and keep only usable unsafe prompts.
--    - prompt_label = 'unsafe': the classifier only sees prompts that
--      detection already flagged.
--    - 'REDACTED' prompts have no text to learn from.
--    - Rows with no category left after cleaning have nothing to predict.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE unsafe_rows AS
WITH stacked AS (
    SELECT 'train' AS split, id, prompt, prompt_label, response, response_label, violated_categories FROM raw_train
    UNION ALL
    SELECT 'validation', id, prompt, prompt_label, response, response_label, violated_categories FROM raw_validation
    UNION ALL
    SELECT 'test', id, prompt, prompt_label, response, response_label, violated_categories FROM raw_test
)
SELECT split, id, trim(prompt) AS prompt, clean_cats(violated_categories) AS cats
FROM stacked
WHERE prompt_label = 'unsafe'
  AND prompt IS NOT NULL
  AND trim(prompt) NOT IN ('', 'REDACTED')
  -- STRICTER OPTION: uncomment to keep only rows where the categories can
  -- only describe the prompt (no response, or the response was safe).
  -- AND (response IS NULL OR response_label = 'safe')
  AND len(clean_cats(violated_categories)) > 0;


-- ---------------------------------------------------------------------
-- 4. Merge duplicate prompts within a split.
--    The same prompt can appear several times (e.g. once with a normal
--    response and once in the refusals file). Keep one row per prompt,
--    with the union of all its categories.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE deduped AS
SELECT split,
       min(id)                              AS id,
       prompt,
       list_sort(list_distinct(flatten(list(cats)))) AS cats
FROM unsafe_rows
GROUP BY split, prompt;


-- ---------------------------------------------------------------------
-- 5. Remove leakage between splits.
--    A prompt in both train and test would let the model be tested on
--    something it already saw, inflating the score. Test wins over
--    validation, and validation wins over train.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE no_leak AS
SELECT d.*
FROM deduped d
WHERE NOT (
        d.split = 'train'
        AND d.prompt IN (SELECT prompt FROM deduped WHERE split IN ('validation', 'test'))
      )
  AND NOT (
        d.split = 'validation'
        AND d.prompt IN (SELECT prompt FROM deduped WHERE split = 'test')
      );


-- ---------------------------------------------------------------------
-- 6. Final category list, counted on TRAIN only
--    (counting on test would let test data influence our choices).
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE categories AS
SELECT cat, count(*) AS n_train
FROM (SELECT unnest(cats) AS cat FROM no_leak WHERE split = 'train')
GROUP BY cat
HAVING count(*) >= (SELECT min_count FROM settings)
ORDER BY n_train DESC;


-- ---------------------------------------------------------------------
-- 7. Drop rare categories from every list, then drop rows left empty.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE prepared AS
WITH kept AS (SELECT list(cat) AS keep FROM categories)
SELECT split, id, prompt,
       list_filter(cats, c -> list_contains(kept.keep, c)) AS cats
FROM no_leak, kept;

DELETE FROM prepared WHERE len(cats) = 0;


-- ---------------------------------------------------------------------
-- 8. One output table per split.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE train_prep      AS SELECT id, prompt, cats FROM prepared WHERE split = 'train';
CREATE OR REPLACE TABLE validation_prep AS SELECT id, prompt, cats FROM prepared WHERE split = 'validation';
CREATE OR REPLACE TABLE test_prep       AS SELECT id, prompt, cats FROM prepared WHERE split = 'test';


-- ---------------------------------------------------------------------
-- 9. Sanity-check summary (the last statement's result is printed).
--    Rows per split, and how many prompts have 1, 2, 3+ categories.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE prep_summary AS
SELECT split,
       count(*)                                    AS prompts,
       count(*) FILTER (WHERE len(cats) = 1)       AS one_category,
       count(*) FILTER (WHERE len(cats) = 2)       AS two_categories,
       count(*) FILTER (WHERE len(cats) >= 3)      AS three_plus
FROM prepared
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END;

SELECT * FROM prep_summary;