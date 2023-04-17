const devRobots = `
User-agent: *
Disallow: /
`;

const prodRobots = `
User-agent: *
Allow: /
`;

export const GET = () => {
	return new Response(import.meta.env.VITE_NODE_ENV === 'production' ? prodRobots : devRobots);
};
