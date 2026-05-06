# Translating EZ-CorridorKey

EZ-CorridorKey uses Qt's built-in translation system. All user-visible strings are marked
for translation and can be exported to standard `.ts` files that work with Qt Linguist.

## How to add a new language

1. Copy the English source file to a new language code:
   ```
   cp corridorkey_en.ts corridorkey_fr.ts
   ```

2. Open the new `.ts` file in [Qt Linguist](https://doc.qt.io/qt-6/linguist-translators.html)
   or edit the XML directly. Each entry has a `<source>` (English) and `<translation>` field.

3. Fill in translations. Mark each entry as "finished" in Qt Linguist when done.

4. Compile the `.ts` file to a binary `.qm` file:
   ```
   pyside6-lrelease corridorkey_fr.ts
   ```
   This creates `corridorkey_fr.qm` in the same directory.

5. Submit a pull request with both the `.ts` and `.qm` files.

## How to update strings after code changes

When new UI strings are added to the code, regenerate the English source:

```bash
pyside6-lupdate -extensions py -recursive ui/ -ts ui/translations/corridorkey_en.ts
```

This updates `corridorkey_en.ts` with any new or changed strings. Existing translations
in other language files are preserved; only new entries are added as untranslated.

To update a specific language file:

```bash
pyside6-lupdate -extensions py -recursive ui/ -ts ui/translations/corridorkey_de.ts
```

## File format

☼ `.ts` files are XML. Translators edit these (with Qt Linguist or a text editor).
☼ `.qm` files are compiled binaries loaded at runtime. Generated from `.ts` files.
☼ Both `.ts` and `.qm` should be committed to the repository.

## Language codes

Use standard ISO 639-1 codes: `en`, `fr`, `de`, `es`, `ja`, `ko`, `zh`, `pt`, `it`, `ru`, etc.

File naming: `corridorkey_{code}.ts` (e.g., `corridorkey_fr.ts` for French).

## Testing your translation

1. Compile your `.ts` to `.qm` (step 4 above).
2. Set the language in EZ-CorridorKey preferences, or set the `LANG` environment variable.
3. Restart the app. Your translation should appear.

## Tips

☼ Keep translations concise. UI space is limited, especially for buttons and labels.
☼ Preserve `%s`, `%d`, `%n` placeholders. These are filled in at runtime with values.
☼ Context names (e.g., "MainWindow", "PreferencesDialog") group strings by where they
  appear in the app. This helps disambiguate identical English strings with different
  meanings in different contexts.
