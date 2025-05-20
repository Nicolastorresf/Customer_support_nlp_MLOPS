ALL_CHAT_WORDS_MAP = {
    'en': {
        'brb': 'be right back',
        'lol': 'laughing out loud',
        'omg': 'oh my god',
        'imo': 'in my opinion',
        'imho': 'in my humble opinion',
        'btw': 'by the way',
        'thx': 'thanks',
        'ty': 'thank you',
        'tyvm': 'thank you very much',
        'np': 'no problem',
        'yw': 'you\'re welcome',
        'dm': 'direct message',
        'pm': 'private message',
        'plz': 'please',
        'u': 'you',
        'r': 'are',
        'y': 'why',
        'c': 'see',
        'k': 'okay',
        'gr8': 'great',
        'l8r': 'later',
        '2moro': 'tomorrow',
        '2nite': 'tonight',
        'asap': 'as soon as possible',
        'fyi': 'for your information',
        'aka': 'also known as',
        'idk': 'i don\'t know',
        'ily': 'i love you',
        'ttyl': 'talk to you later',
        'hbd': 'happy birthday',
        'irl': 'in real life',
        'smh': 'shaking my head',
        'ftw': 'for the win',
        'wtf': 'what the f*ck', # Considerar si se quiere la expansión censurada o no
        'wth': 'what the hell',
        'jk': 'just kidding',
        'icymi': 'in case you missed it',
        'tldr': 'too long; didn\'t read',
        'bff': 'best friends forever',
        'fomo': 'fear of missing out',
        'yolo': 'you only live once',
        'tbh': 'to be honest',
        'ikr': 'i know, right?',
        'afaik': 'as far as i know',
        'afk': 'away from keyboard',
        'gg': 'good game',
        'gj': 'good job',
        'gl': 'good luck',
        'hf': 'have fun',
        'rn': 'right now',
        'omw': 'on my way',
        'sup': 'what\'s up',
        'wbu': 'what about you',
        'hbu': 'how about you',
        'nvm': 'never mind',
        'wyd': 'what are you doing',
        'idc': 'i don\'t care',
        'fml': 'f*ck my life', # Considerar si se quiere la expansión censurada o no
        'cya': 'see you',
        'cu': 'see you',
        'cul8r': 'see you later'
    },
    'es': {
        'tqm': 'te quiero mucho',
        'tkm': 'te quiero mucho', # Normalizado a una expansión común
        'pq': 'porque',
        'xq': 'por qué', # Diferenciar de 'pq' si es necesario semánticamente
        'x': 'por',
        'q': 'que',
        'k': 'que', # 'k' puede ser 'que' o 'okey', contexto dependiente. Se elige 'que'.
        'd': 'de',
        'finde': 'fin de semana',
        'ntp': 'no te preocupes',
        'npn': 'no pasa nada',
        'mdi': 'me da igual',
        'maso': 'más o menos', # Añadir tilde
        'xfi': 'por fin',
        'xfa': 'por favor',
        'porfa': 'por favor',
        'salu2': 'saludos',
        'bss': 'besos',
        'ctm': 'concha tu madre', # Vulgar, mantener o no según política de datos
        'wey': 'güey', # Común en México
        'parce': 'amigo', # Común en Colombia
        'chido': 'genial', # Común en México
        'bacano': 'genial', # Común en Colombia y otros
        'chévere': 'genial', # Común en varios países de Latam
        'chale': 'expresión de resignación o decepción', # Común en México
        'neta': 'la verdad', # Común en México
        'chamba': 'trabajo', # Común en Latam
        'chavo': 'muchacho', # Común en México
        'chava': 'muchacha', # Común en México
        'vrdd': 'verdad',
        'bn': 'bien',
        'hla': 'hola',
        'grax': 'gracias',
        'dnd': 'de nada',
        'pls': 'por favor', # Anglicismo
        'wapo': 'guapo',
        'wapa': 'guapa'
    },
    'fr': {
        'mdr': 'mort de rire',
        'ptdr': 'pété de rire', # Vulgar/Informal
        'lol': 'laughing out loud', # Anglicismo común
        'slt': 'salut',
        'bjr': 'bonjour',
        'bsr': 'bonsoir',
        'stp': 's\'il te plaît',
        'svp': 's\'il vous plaît',
        'cad': 'c\'est-à-dire',
        'auj': 'aujourd\'hui',
        'dem': 'demain',
        'rdv': 'rendez-vous',
        'qqn': 'quelqu\'un',
        'qqch': 'quelque chose',
        'bcp': 'beaucoup',
        'pk': 'pourquoi',
        'pck': 'parce que',
        'jtm': 'je t\'aime',
        'dsl': 'désolé',
        'tg': 'ta gueule' # Vulgar
    },
    'de': {
        'lol': 'laughing out loud', # Anglicismo común
        'omg': 'oh mein Gott', # Anglicismo común
        'hdl': 'hab dich lieb',
        'ida': 'ich dich auch',
        'ka': 'keine Ahnung', # Normalizado desde 'kA'
        'vlt': 'vielleicht',
        'evtl': 'eventuell',
        'bzw': 'beziehungsweise',
        'usw': 'und so weiter',
        'zb': 'zum Beispiel', # Normalizado desde 'zB'
        'mfg': 'Mit freundlichen Grüßen', # Normalizado desde 'MfG'
        'lg': 'Liebe Grüße', # Normalizado desde 'LG'
        'we': 'Wochenende', # Normalizado desde 'WE'
        'gn8': 'gute Nacht',
        'kd': 'kein Ding', # Normalizado desde 'kD'
        'pls': 'bitte', # Anglicismo
        'thx': 'danke', # Anglicismo
        'cu': 'see you', # Anglicismo
        'wmd': 'was machst du'
    },
    'pt': {
        'vc': 'você',
        'fds': 'fim de semana',
        'tb': 'também',
        'tbm': 'também',
        'blz': 'beleza',
        'flw': 'falou', # Gíria para despedida, ok
        'vlw': 'valeu', # Gíria para obrigado/ok
        'sdds': 'saudades',
        'abs': 'abraços',
        'bjs': 'beijos',
        'bj': 'beijo',
        'obg': 'obrigado', # Usar 'obrigado(a)' o manejar género en post-procesamiento si es necesario
        'obgda': 'obrigada',
        'pfv': 'por favor',
        'cmg': 'comigo',
        'ctg': 'contigo',
        'q': 'que',
        'pra': 'para',
        'hj': 'hoje',
        'agr': 'agora',
        'vdd': 'verdade',
        'sqn': 'só que não',
        'namo': 'namorado_namorada' # Neutralizado o especificar, e.g., "namorado ou namorada"
    },
    'it': {
        'cmq': 'comunque',
        'tvb': 'ti voglio bene',
        'tvtb': 'ti voglio tanto bene',
        'nn': 'non',
        'prg': 'prego',
        'grz': 'grazie',
        'pfv': 'per favore',
        'xke': 'perché',
        'xò': 'però',
        'ke': 'che',
        'qlcs': 'qualcosa',
        'qlcn': 'qualcuno',
        'tt': 'tutto',
        'bn': 'bene',
        'dp': 'dopo',
        'msg': 'messaggio',
        'risp': 'rispondi'
    },
}    