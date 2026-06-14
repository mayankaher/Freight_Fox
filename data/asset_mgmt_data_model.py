import graphviz
from graphviz import Digraph
import os

# Create output directory
output_dir = '/Users/mayankaher/Desktop/freightfox/data/output'
os.makedirs(output_dir, exist_ok=True)

# Initialize graph with purple theme matching Salesforce style
dot = Digraph(
    engine='dot',
    name='AssetMgmt_DataModel',
    format='png',
    graph_attr={
        'rankdir': 'LR',
        'bgcolor': 'white',
        'fontname': 'Helvetica,Arial,sans-serif',
        'fontsize': '36',
        'nodesep': '0.8',
        'ranksep': '2.0',
        'splines': 'spline',
        'overlap': 'false',
        'pad': '1.0',
        'newrank': 'true',
        'concentrate': 'true',
        'dpi': '300'
    },
    node_attr={
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#F3E5F5',  # Light purple
        'color': '#4A148C',       # Deep purple border
        'fontname': 'Helvetica,Arial,sans-serif',
        'fontsize': '18',
        'fontcolor': '#4A148C',
        'width': '4.0',
        'height': '1.2'
    },
    edge_attr={
        'color': '#7B1FA2',
        'fontname': 'Helvetica,Arial,sans-serif',
        'fontsize': '16',
        'fontcolor': '#4A148C',
        'arrowhead': 'vee',
        'arrowsize': '1.0'
    }
)

# Define Subject Area clusters with their entities and attributes
clusters = {
    'Client_Investor': {
        'label': 'Client & Investor Mgmt',
        'color': '#E1BEE7',
        'entities': {
            'INVESTOR': 'INVESTOR\n• Individual/Institution\n• Committing capital',
            'CLIENT_ACCOUNT': 'CLIENT ACCOUNT\n• Book-of-record account\n• Links to holdings',
            'KYC_RECORD': 'KYC RECORD\n• Verified identity\n• Due-diligence record',
            'IDENTITY_DOCUMENT': 'IDENTITY DOCUMENT\n• Passport, utility bill\n• Evidence for KYC',
            'MANDATE': 'MANDATE\n• Governing investment\n• Account mandate',
            'SUBSCRIPTION_REDEMPTION': 'SUBSCRIPTION / REDEMPTION\n• Investor transactions\n• In/out of share classes',
            'AML_ALERT': 'AML ALERT\n• Suspicious activity flag\n• System-generated'
        }
    },
    'Fund_Product': {
        'label': 'Fund & Product',
        'color': '#D1C4E9',
        'entities': {
            'FUND': 'FUND\n• Collective investment vehicle\n• UCITS, AIF, SMA',
            'SHARE_CLASS': 'SHARE CLASS\n• Tranche of fund\n• Fee/currency attributes',
            'FUND_NAV': 'FUND NAV\n• Per-share net asset value\n• At valuation point',
            'FUND_EXPENSE': 'FUND EXPENSE\n• Ongoing charges\n• Mgmt/custody/audit fees',
            'FUND_DISCLOSURE': 'FUND DISCLOSURE\n• KIID, PRIIPs KID\n• SFDR disclosure',
            'BENCHMARK': 'BENCHMARK\n• Reference index\n• Performance comparison'
        }
    },
    'Portfolio_Mgmt': {
        'label': 'Portfolio Management',
        'color': '#C5CAE9',
        'entities': {
            'PORTFOLIO': 'PORTFOLIO\n• Managed collection\n• Of securities',
            'POSITION': 'POSITION\n• Holding of security\n• Within portfolio',
            'INVESTMENT_RESTRICTION': 'INVESTMENT RESTRICTION\n• Pre/post-trade rules\n• Allowable exposures',
            'PERFORMANCE_RECORD': 'PERFORMANCE RECORD\n• Time-weighted return\n• Attribution records'
        }
    },
    'Trade_Order': {
        'label': 'Trade & Order Mgmt',
        'color': '#BBDEFB',
        'entities': {
            'ORDER': 'ORDER\n• Instruction to transact\n• Buy/sell security',
            'TRADE_EXECUTION': 'TRADE / EXECUTION\n• Confirmed transaction\n• Buyer/seller',
            'ALLOCATION': 'ALLOCATION\n• Portion of trade\n• Assigned to portfolio',
            'SETTLEMENT_INSTRUCTION': 'SETTLEMENT INSTRUCTION\n• Custodian-directed\n• Deliver/receive assets'
        }
    },
    'Securities_Ref': {
        'label': 'Securities & Ref Data',
        'color': '#B2DFDB',
        'entities': {
            'SECURITY': 'SECURITY\n• Tradeable instrument\n• Equity, bond, derivative',
            'ISSUER': 'ISSUER\n• Legal entity\n• Creates security',
            'PRICE': 'PRICE\n• Time-series market prices\n• Bid, ask, last, close',
            'CORPORATE_ACTION': 'CORPORATE ACTION\n• Events altering terms\n• Dividends, splits, mergers'
        }
    },
    'Risk_Mgmt': {
        'label': 'Risk Management',
        'color': '#B2EBF2',
        'entities': {
            'RISK_METRIC': 'RISK METRIC\n• VaR, beta, duration\n• Tracking error',
            'RISK_LIMIT': 'RISK LIMIT\n• Threshold governing\n• Allowable exposure',
            'COUNTERPARTY_EXPOSURE': 'COUNTERPARTY EXPOSURE\n• Net exposure\n• Across trades',
            'LIQUIDITY_RISK_RECORD': 'LIQUIDITY RISK RECORD\n• Fund redemption ability\n• Under stress'
        }
    },
    'Compliance': {
        'label': 'Compliance & Regulatory',
        'color': '#B3E5FC',
        'entities': {
            'REGULATORY_REPORT': 'REGULATORY REPORT\n• Structured submission\n• MiFID II, EMIR, AIFMD',
            'REGULATION': 'REGULATION\n• Governing rule\n• For report',
            'RESTRICTION_BREACH': 'RESTRICTION BREACH\n• Violation recorded\n• Of investment rule',
            'ESG_SCORE': 'ESG SCORE\n• E, S, G pillars\n• From providers',
            'SFDR_DISCLOSURE': 'SFDR DISCLOSURE\n• Article 8 or 9\n• Product-level report'
        }
    },
    'Fund_Accounting': {
        'label': 'Fund Accounting & Ops',
        'color': '#DCEDC8',
        'entities': {
            'FUND_ACCOUNTING_RECORD': 'FUND ACCOUNTING RECORD\n• Double-entry record\n• For all transactions',
            'CASH_MOVEMENT': 'CASH MOVEMENT\n• Debit/credit of cash\n• In fund account',
            'CUSTODY_ACCOUNT': 'CUSTODY ACCOUNT\n• Safekeeping account\n• Holds fund assets',
            'FEE_BILLING_RECORD': 'FEE / BILLING RECORD\n• Invoice/fee statement\n• To client account'
        }
    },
    'Counterparty_Market': {
        'label': 'Counterparty & Market',
        'color': '#F0F4C3',
        'entities': {
            'COUNTERPARTY_BROKER': 'COUNTERPARTY / BROKER\n• Executes trades\n• On behalf of fund',
            'CUSTODIAN': 'CUSTODIAN\n• Holds/safeguards\n• Fund assets',
            'EXCHANGE_VENUE': 'EXCHANGE / VENUE\n• Regulated market\n• MTF trading'
        }
    },
    'Enterprise_Party': {
        'label': 'Enterprise & Party',
        'color': '#FFE0B2',
        'entities': {
            'LEGAL_ENTITY': 'LEGAL ENTITY\n• Corporation/partnership\n• Natural person',
            'EMPLOYEE_ADVISOR': 'EMPLOYEE / ADVISOR\n• Employed by/advising\n• Asset manager',
            'ROLE': 'ROLE\n• PM, RM, compliance\n• Officer over time',
            'CONTRACT_AGREEMENT': 'CONTRACT / AGREEMENT\n• Legally binding\n• Between parties'
        }
    },
    'Distribution': {
        'label': 'Distribution & Sales',
        'color': '#FFCCBC',
        'entities': {
            'DISTRIBUTOR_IFA': 'DISTRIBUTOR / IFA\n• Bank, platform, IFA\n• Distributing products',
            'DISTRIBUTION_AGREEMENT': 'DISTRIBUTION AGREEMENT\n• Commercial terms\n• Fund distribution',
            'TRANSFER_AGENCY_RECORD': 'TRANSFER AGENCY RECORD\n• TA\'s record\n• Subscription/redemption'
        }
    },
    'Finance': {
        'label': 'Finance & Reporting',
        'color': '#F8BBD0',
        'entities': {
            'GENERAL_LEDGER_ENTRY': 'GENERAL LEDGER ENTRY\n• Double-entry record\n• Chart of accounts',
            'COST_CENTER': 'COST CENTER\n• Organisational unit\n• Financial reporting',
            'BUDGET': 'BUDGET\n• Annual budget\n• By expense category',
            'REVENUE_RECORD': 'REVENUE RECORD\n• Mgmt/performance fee\n• Revenue earned'
        }
    },
    'Sustainability': {
        'label': 'Sustainability & ESG',
        'color': '#E6EE9C',
        'entities': {
            'ESG_TARGET': 'ESG TARGET\n• Sustainability goal\n• Paris-alignment',
            'CARBON_FOOTPRINT': 'CARBON FOOTPRINT\n• GHG emissions\n• Scope 1, 2, 3'
        }
    }
}

# Create clusters
for cluster_id, cluster_data in clusters.items():
    with dot.subgraph(name=f'cluster_{cluster_id}') as c:
        c.attr(
            label=cluster_data['label'],
            style='rounded,filled',
            fillcolor=cluster_data['color'],
            color='#4A148C',
            fontname='Helvetica,Arial,sans-serif',
            fontsize='24',
            fontcolor='#4A148C',
            penwidth='3'
        )
        for entity_id, entity_label in cluster_data['entities'].items():
            c.node(entity_id, entity_label)

# Define relationships: (from, to, label, cardinality_from, cardinality_to, style)
relationships = [
    # Client & Investor Mgmt
    ('INVESTOR', 'CLIENT_ACCOUNT', 'holds', '1', 'N', 'Compositional'),
    ('INVESTOR', 'KYC_RECORD', 'has', '1', '1', 'Attributive'),
    ('INVESTOR', 'SUBSCRIPTION_REDEMPTION', 'initiates', '1', 'N', 'Associative'),
    ('KYC_RECORD', 'IDENTITY_DOCUMENT', 'evidenced by', '1', 'N', 'Compositional'),
    ('CLIENT_ACCOUNT', 'MANDATE', 'operates under', '1', '1', 'Attributive'),
    ('CLIENT_ACCOUNT', 'PORTFOLIO', 'associated with', '1', 'N', 'Associative'),
    ('INVESTOR', 'LEGAL_ENTITY', 'instance of', 'N', '1', 'Associative'),
    ('AML_ALERT', 'INVESTOR', 'triggered by', 'N', '1', 'Causal'),

    # Fund & Product
    ('FUND', 'SHARE_CLASS', 'issues', '1', 'N', 'Compositional'),
    ('FUND', 'FUND_NAV', 'has', '1', 'N', 'Attributive'),
    ('FUND', 'BENCHMARK', 'references', 'N', '1', 'Associative'),
    ('FUND', 'LEGAL_ENTITY', 'constituted as', 'N', '1', 'Associative'),
    ('FUND', 'FUND_EXPENSE', 'accrues', '1', 'N', 'Attributive'),
    ('FUND', 'FUND_DISCLOSURE', 'produces', '1', 'N', 'Attributive'),
    ('SHARE_CLASS', 'SUBSCRIPTION_REDEMPTION', 'subscribed to', '1', 'N', 'Associative'),
    ('FUND_NAV', 'PRICE', 'inputs from', 'M', 'N', 'Causal'),

    # Portfolio Management
    ('PORTFOLIO', 'POSITION', 'composed of', '1', 'N', 'Compositional'),
    ('PORTFOLIO', 'INVESTMENT_RESTRICTION', 'constrained by', '1', 'N', 'Compositional'),
    ('PORTFOLIO', 'PERFORMANCE_RECORD', 'produces', '1', 'N', 'Attributive'),
    ('PORTFOLIO', 'BENCHMARK', 'measured against', 'N', '1', 'Associative'),
    ('POSITION', 'SECURITY', 'references', 'N', '1', 'Associative'),
    ('POSITION', 'RISK_METRIC', 'calculated at', '1', 'N', 'Attributive'),
    ('PERFORMANCE_RECORD', 'BENCHMARK', 'attributed to', 'N', '1', 'Associative'),
    ('PORTFOLIO', 'FUND', 'services/spans', 'M', 'N', 'Associative'),

    # Trade & Order Mgmt
    ('ORDER', 'TRADE_EXECUTION', 'results in', '1', 'N', 'Causal'),
    ('TRADE_EXECUTION', 'ALLOCATION', 'allocated to', '1', 'N', 'Compositional'),
    ('ALLOCATION', 'SETTLEMENT_INSTRUCTION', 'generates', '1', '1', 'Transformational'),
    ('ALLOCATION', 'POSITION', 'updates', 'M', 'N', 'Transitive'),
    ('ORDER', 'PORTFOLIO', 'generated for', 'N', '1', 'Associative'),
    ('TRADE_EXECUTION', 'COUNTERPARTY_BROKER', 'executed with', 'N', '1', 'Associative'),
    ('TRADE_EXECUTION', 'SECURITY', 'references', 'N', '1', 'Associative'),
    ('SETTLEMENT_INSTRUCTION', 'CASH_MOVEMENT', 'results in', '1', '1', 'Transformational'),

    # Securities & Ref Data
    ('SECURITY', 'ISSUER', 'issued by', 'N', '1', 'Associative'),
    ('SECURITY', 'PRICE', 'has', '1', 'N', 'Attributive'),
    ('SECURITY', 'CORPORATE_ACTION', 'has', '1', 'N', 'Causal'),
    ('CORPORATE_ACTION', 'POSITION', 'adjusts', '1', 'N', 'Causal'),
    ('SECURITY', 'ESG_SCORE', 'assigned', '1', 'N', 'Attributive'),
    ('SECURITY', 'BENCHMARK', 'constituent of', 'M', 'N', 'Compositional'),
    ('ISSUER', 'LEGAL_ENTITY', 'resolves to', 'N', '1', 'Associative'),

    # Risk Management
    ('RISK_METRIC', 'PORTFOLIO', 'rolls up to', 'N', '1', 'Associative'),
    ('RISK_METRIC', 'RISK_LIMIT', 'governed by', 'N', '1', 'Attributive'),
    ('RISK_LIMIT', 'INVESTMENT_RESTRICTION', 'operationalised as', '1', '1', 'Attributive'),
    ('RISK_METRIC', 'SECURITY', 'references', 'N', '1', 'Associative'),
    ('COUNTERPARTY_EXPOSURE', 'COUNTERPARTY_BROKER', 'calculated against', 'N', '1', 'Attributive'),
    ('LIQUIDITY_RISK_RECORD', 'FUND', 'produced at', 'N', '1', 'Attributive'),

    # Compliance & Regulatory
    ('REGULATORY_REPORT', 'REGULATION', 'satisfies', 'N', '1', 'Associative'),
    ('ESG_SCORE', 'SECURITY', 'assigned to', 'N', '1', 'Attributive'),
    ('ESG_SCORE', 'FUND', 'aggregated at', 'N', '1', 'Attributive'),
    ('FUND_DISCLOSURE', 'FUND', 'produced per', 'N', '1', 'Associative'),
    ('RESTRICTION_BREACH', 'INVESTMENT_RESTRICTION', 'violates', 'N', '1', 'Causal'),
    ('RESTRICTION_BREACH', 'TRADE_EXECUTION', 'caused by', '1', '1', 'Causal'),
    ('AML_ALERT', 'INVESTOR', 'flagged on', 'N', '1', 'Causal'),

    # Fund Accounting & Ops
    ('FUND_ACCOUNTING_RECORD', 'FUND', 'belongs to', 'N', '1', 'Associative'),
    ('FUND_NAV', 'FUND_ACCOUNTING_RECORD', 'computed from', '1', '1', 'Causal'),
    ('CASH_MOVEMENT', 'FUND_ACCOUNTING_RECORD', 'posted as', 'N', '1', 'Transitive'),
    ('FUND_EXPENSE', 'FUND', 'accrued against', 'N', '1', 'Attributive'),
    ('CUSTODY_ACCOUNT', 'FUND', 'holds assets for', 'N', '1', 'Associative'),
    ('CUSTODY_ACCOUNT', 'CUSTODIAN', 'maintained at', 'N', '1', 'Associative'),
    ('FEE_BILLING_RECORD', 'CLIENT_ACCOUNT', 'billed to', 'N', '1', 'Attributive'),
    ('FUND_ACCOUNTING_RECORD', 'GENERAL_LEDGER_ENTRY', 'generates', '1', 'N', 'Transformational'),

    # Counterparty & Market
    ('COUNTERPARTY_BROKER', 'TRADE_EXECUTION', 'executes', '1', 'N', 'Associative'),
    ('CUSTODIAN', 'CUSTODY_ACCOUNT', 'maintains', '1', 'N', 'Compositional'),
    ('EXCHANGE_VENUE', 'TRADE_EXECUTION', 'hosts', '1', 'N', 'Associative'),
    ('COUNTERPARTY_BROKER', 'COUNTERPARTY_EXPOSURE', 'monitored for', '1', 'N', 'Attributive'),

    # Enterprise & Party
    ('LEGAL_ENTITY', 'LEGAL_ENTITY', 'parent of', '1', 'N', 'Compositional'),
    ('LEGAL_ENTITY', 'CONTRACT_AGREEMENT', 'party to', '1', 'N', 'Associative'),
    ('EMPLOYEE_ADVISOR', 'CLIENT_ACCOUNT', 'assigned to', 'M', 'N', 'Associative'),
    ('EMPLOYEE_ADVISOR', 'ROLE', 'holds', 'M', 'N', 'Attributive'),
    ('CONTRACT_AGREEMENT', 'FUND', 'governs', 'M', 'N', 'Associative'),

    # Distribution & Sales
    ('DISTRIBUTOR_IFA', 'DISTRIBUTION_AGREEMENT', 'operates under', '1', 'N', 'Compositional'),
    ('DISTRIBUTION_AGREEMENT', 'FUND', 'covers', 'M', 'N', 'Associative'),
    ('TRANSFER_AGENCY_RECORD', 'SUBSCRIPTION_REDEMPTION', 'records', '1', '1', 'Transformational'),
    ('DISTRIBUTOR_IFA', 'LEGAL_ENTITY', 'instance of', 'N', '1', 'Associative'),

    # Finance & Reporting
    ('GENERAL_LEDGER_ENTRY', 'COST_CENTER', 'attributed to', 'N', '1', 'Attributive'),
    ('COST_CENTER', 'BUDGET', 'has', '1', 'N', 'Attributive'),
    ('GENERAL_LEDGER_ENTRY', 'FUND_ACCOUNTING_RECORD', 'derived from', 'N', '1', 'Transitive'),
    ('REVENUE_RECORD', 'FUND', 'recognised at', 'N', '1', 'Attributive'),

    # Sustainability & ESG
    ('ESG_TARGET', 'FUND', 'set at', 'N', '1', 'Attributive'),
    ('ESG_TARGET', 'ESG_SCORE', 'measured by', '1', 'N', 'Attributive'),
    ('SFDR_DISCLOSURE', 'REGULATORY_REPORT', 'is type of', 'N', '1', 'Causal'),
    ('CARBON_FOOTPRINT', 'PORTFOLIO', 'calculated at', 'N', '1', 'Attributive'),
    ('CARBON_FOOTPRINT', 'ISSUER', 'attributed to', 'N', '1', 'Associative')
]

# Color mapping for relationship forms
relationship_colors = {
    'Associative': '#7B1FA2',
    'Compositional': '#E65100',
    'Attributive': '#1565C0',
    'Causal': '#2E7D32',
    'Transitive': '#6A1B9A',
    'Transformational': '#C62828'
}

# Add edges with labels
for from_entity, to_entity, label, card_from, card_to, rel_form in relationships:
    edge_label = f'{label}\n[{card_from}:{card_to}] {rel_form[:4]}'
    edge_color = relationship_colors.get(rel_form, '#7B1FA2')

    # Style based on relationship form
    if rel_form == 'Compositional':
        style = 'solid'
        penwidth = '2'
    elif rel_form == 'Causal':
        style = 'dashed'
        penwidth = '1.5'
    elif rel_form == 'Transformational':
        style = 'dotted'
        penwidth = '1.5'
    elif rel_form == 'Transitive':
        style = 'dashed'
        penwidth = '1'
    else:
        style = 'solid'
        penwidth = '1'

    dot.edge(
        from_entity,
        to_entity,
        xlabel=edge_label,   # xlabel works with all spline types
        color=edge_color,
        fontcolor=edge_color,
        style=style,
        penwidth=penwidth
    )

# Add title
dot.attr(
    label='\n\nAsset Management Enterprise Data Taxonomy\nMaster Relationship Model\n\n',
    labelloc='t',
    fontsize='48',
    fontcolor='#4A148C',
    fontname='Helvetica,Arial,sans-serif'
)

# ── Legend subgraph ──────────────────────────────────────────────────────────
with dot.subgraph(name='cluster_legend') as leg:
    leg.attr(
        label='Legend — Relationship Forms',
        style='rounded,filled',
        fillcolor='#FAFAFA',
        color='#4A148C',
        fontname='Helvetica,Arial,sans-serif',
        fontsize='20',
        fontcolor='#4A148C',
        penwidth='2'
    )
    legend_items = [
        ('leg_assoc',  'Associative  (solid, purple)',   '#7B1FA2', 'solid',  '1'),
        ('leg_comp',   'Compositional (solid, orange)',  '#E65100', 'solid',  '2'),
        ('leg_attr',   'Attributive  (solid, blue)',     '#1565C0', 'solid',  '1'),
        ('leg_causal', 'Causal       (dashed, green)',   '#2E7D32', 'dashed', '1.5'),
        ('leg_trans',  'Transitive   (dashed, violet)',  '#6A1B9A', 'dashed', '1'),
        ('leg_transf', 'Transformational (dotted, red)', '#C62828', 'dotted', '1.5'),
    ]
    # invisible anchor nodes for each legend row
    prev = None
    for nid, nlabel, color, lstyle, pw in legend_items:
        leg.node(nid, nlabel,
                 shape='plaintext',
                 style='',
                 fillcolor='white',
                 fontcolor=color,
                 fontname='Helvetica,Arial,sans-serif',
                 fontsize='18',
                 width='4', height='0.5')
        if prev:
            leg.edge(prev, nid, style='invis')
        prev = nid

# Render
output_path = os.path.join(output_dir, 'AssetMgmt_DataModel')
dot.render(output_path, cleanup=True, format='png')

print(f"Diagram saved to: {output_path}.png")
print(f"Total entities: {sum(len(c['entities']) for c in clusters.values())}")
print(f"Total relationships: {len(relationships)}")
