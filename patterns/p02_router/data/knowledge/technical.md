# Ledgerly — Technical knowledge base

## Supported platforms
- Web: the latest two versions of Chrome, Firefox, Safari and Edge.
- Mobile apps: iOS 16 or later, Android 10 or later. Older versions are not supported.

## Offline editing and sync
Invoices drafted offline in the mobile app sync when the device reconnects. If the same invoice was edited on two devices, the last edit wins; the previous version is kept in the invoice's version history.

## Integrations
- **Stripe and PayPal** for collecting payments. Payments collected through Stripe are recorded on the invoice automatically within 5 minutes. If they do not appear, reconnect Stripe in Settings > Integrations.
- **QuickBooks Online and Xero** (Pro and Team plans only). Data syncs every 15 minutes. QuickBooks Desktop is not supported.
- **Zapier** for custom automations.

## Importing and exporting
- Client import via CSV: maximum 5,000 rows per file, UTF-8 encoding, required columns `name` and `email`. Larger lists must be split into several files.
- Invoices can be exported individually as PDF or in bulk as CSV.
- A full data export (ZIP) can be requested in Settings > Data > Export. It is emailed within 24 hours.

## Invoice emails
- Invoices are sent from invoices@ledgerly.app.
- If a client does not receive an invoice: ask them to check spam, verify the client's email address in Ledgerly, and resend. Bounced emails are marked with a red badge on the invoice.
- Team plan only: invoices can be sent from your own domain after adding SPF and DKIM DNS records in Settings > Email.

## Invoice PDFs
Logos must be PNG or JPG, up to 2 MB. The recommended size is 600x200 pixels; smaller images can look blurry.

## API
REST API available on Pro and Team plans. Rate limit: 100 requests per minute. API keys are created in Settings > Developers.

## Troubleshooting
Check status.ledgerly.app for ongoing incidents. For display problems, clear the browser cache and disable browser extensions.