/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		fontFamily: {
			sans: ['Poppins', 'sans-serif']
		},

		borderRadius: {
			none: '0',
			sm: '4px',
			md: '4px',
			lg: '10px',
			full: '16px',
			large: '24px',
			max: '9999px'
		},
		extend: {
			screens: {
				'3xl': '1600px'
			},
			borderWidth: {
				1: '1px'
			},
			colors: {
				'll-primary': '#061E30',
				'll-accent': '#ED5A55',
				'll-supportive3': '#0C3451',
				'll-accent2': '#EBF4F7',
				'll-accent3': '#11BBBB',

				'll-p': 'rgb(7,73,117)',
				'll-p-dark': 'rgb(19,53,82)',
				'll-p-light': 'rgb(217,239,250)',
				'll-p-x-light': 'rgb(236,247,252)',

				'll-s-1': 'rgb(195,60,102)',
				'll-s-1-signal': 'rgb(231,43,110)',

				'll-s-2': 'rgb(94,78,151)',
				'll-s-2-light': 'rgb(155,131,188)',

				'll-t-1': 'rgb(233,124,46)',
				'll-t-1-signal': 'rgb(248,149,58)',

				'll-t-2': 'rgb(118,90,68)',
				'll-t-2-light': 'rgb(166,115,75)',

				'll-b-1': 'rgb(248,232,232)',
				'll-b-2': 'rgb(239,232,245)',
				'll-b-3': 'rgb(249,230,217)',

				'll-black-1': 'rgb(26,26,26)',
				'll-grey-1': 'rgb(232,232,232)',
				'll-grey-2': '#686868',
				'll-link-1': 'rgb(0,102,165)'
			}
		},
		plugins: []
	},
	plugins: [require('daisyui')]
};
