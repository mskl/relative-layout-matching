# 11 fields + 1 unk class
SELECTED_FIELDS = (
	'amount_total', 'bank_num', 'date_issue', 'document_id', 'phone_num',
	'recipient_address', 'recipient_name', 'sender_address', 'sender_dic',
	'sender_ic', 'sender_name'
)

FIELD2CLASS = dict(zip(SELECTED_FIELDS, range(len(SELECTED_FIELDS))))
